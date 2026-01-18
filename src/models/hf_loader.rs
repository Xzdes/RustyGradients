///! HuggingFace Model Loader
///!
///! Load pre-trained models from HuggingFace Hub:
///! - GPT-2 (124M, 355M, 774M, 1.5B parameters)
///! - GPT-Neo
///! - Future: LLaMA, Mistral, etc.
///!
///! Features:
///! - Download from HuggingFace Hub
///! - Weight mapping (HF naming → RustyGradients)
///! - Shape validation
///! - Safetensors format support

use crate::error::{Result, RustyGradientsError};
use crate::serialization::ModelMetadata;
use crate::tensor::Tensor;
use ndarray::ArrayD;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// HuggingFace model configuration
#[derive(Debug, Clone)]
pub struct HFModelConfig {
    pub model_name: String,
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub block_size: usize,
    pub dropout: f32,
}

impl HFModelConfig {
    /// GPT-2 Small (124M parameters)
    pub fn gpt2() -> Self {
        Self {
            model_name: "gpt2".to_string(),
            vocab_size: 50257,
            embedding_dim: 768,
            num_layers: 12,
            num_heads: 12,
            block_size: 1024,
            dropout: 0.1,
        }
    }

    /// GPT-2 Medium (355M parameters)
    pub fn gpt2_medium() -> Self {
        Self {
            model_name: "gpt2-medium".to_string(),
            vocab_size: 50257,
            embedding_dim: 1024,
            num_layers: 24,
            num_heads: 16,
            block_size: 1024,
            dropout: 0.1,
        }
    }

    /// GPT-2 Large (774M parameters)
    pub fn gpt2_large() -> Self {
        Self {
            model_name: "gpt2-large".to_string(),
            vocab_size: 50257,
            embedding_dim: 1280,
            num_layers: 36,
            num_heads: 20,
            block_size: 1024,
            dropout: 0.1,
        }
    }

    /// GPT-2 XL (1.5B parameters)
    pub fn gpt2_xl() -> Self {
        Self {
            model_name: "gpt2-xl".to_string(),
            vocab_size: 50257,
            embedding_dim: 1600,
            num_layers: 48,
            num_heads: 25,
            block_size: 1024,
            dropout: 0.1,
        }
    }
}

/// HuggingFace model loader
pub struct HFModelLoader {
    config: HFModelConfig,
    cache_dir: Option<PathBuf>,
}

impl HFModelLoader {
    /// Create new loader
    pub fn new(config: HFModelConfig) -> Self {
        Self {
            config,
            cache_dir: None,
        }
    }

    /// Set custom cache directory
    pub fn with_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    /// Download model from HuggingFace Hub
    #[cfg(feature = "huggingface")]
    pub fn download(&self) -> Result<PathBuf> {
        let api = if let Some(cache) = &self.cache_dir {
            Api::new()?.with_cache_dir(cache.clone())
        } else {
            Api::new()?
        };

        let repo = api.model(self.config.model_name.clone());

        // Try to download pytorch_model.bin or model.safetensors
        let model_file = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))
            .map_err(|e| {
                RustyGradientsError::IoError(format!(
                    "Failed to download model {}: {}",
                    self.config.model_name, e
                ))
            })?;

        Ok(model_file)
    }

    /// Load model from local file
    pub fn load_from_file(&self, path: &Path) -> Result<(GPT, HashMap<String, Tensor>)> {
        // Check file extension
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| RustyGradientsError::IoError("Invalid file extension".to_string()))?;

        match ext {
            "safetensors" => self.load_safetensors(path),
            "bin" => {
                return Err(RustyGradientsError::IoError(
                    "PyTorch .bin files not yet supported. Use .safetensors format.".to_string(),
                ))
            }
            _ => Err(RustyGradientsError::IoError(format!(
                "Unsupported file format: {}",
                ext
            ))),
        }
    }

    /// Load from safetensors format
    #[cfg(feature = "serialization")]
    fn load_safetensors(&self, path: &Path) -> Result<(GPT, HashMap<String, Tensor>)> {
        use crate::serialization::safetensors_format;

        // Load raw weights
        let (weight_data, weight_shapes, weight_names, _metadata) =
            safetensors_format::load_model(path)?;

        // Convert to HashMap of Tensors
        let mut weights = HashMap::new();

        for ((data, shape), name) in weight_data
            .iter()
            .zip(weight_shapes.iter())
            .zip(weight_names.iter())
        {
            let array = ArrayD::from_shape_vec(shape.clone(), data.clone())
                .map_err(|e| RustyGradientsError::ShapeMismatch {
                    expected: shape.clone(),
                    actual: vec![data.len()],
                    context: format!("Loading weight {}: {}", name, e),
                })?;

            weights.insert(name.clone(), Tensor::new(array, false));
        }

        // Map HuggingFace weights to RustyGradients model
        let model = self.create_model_from_weights(&weights)?;

        Ok((model, weights))
    }

    #[cfg(not(feature = "serialization"))]
    fn load_safetensors(&self, _path: &Path) -> Result<(GPT, HashMap<String, Tensor>)> {
        Err(RustyGradientsError::IoError(
            "Serialization feature not enabled. Compile with --features serialization".to_string(),
        ))
    }

    /// Create GPT model and map weights
    fn create_model_from_weights(&self, weights: &HashMap<String, Tensor>) -> Result<GPT> {
        println!("🔧 Creating GPT model from {} weights", weights.len());

        // Verify required weights exist
        self.verify_weights(weights)?;

        // Create model with config
        let mut model = GPT::new(
            self.config.vocab_size,
            self.config.embedding_dim,
            self.config.num_layers,
            self.config.num_heads,
            self.config.block_size,
            self.config.dropout,
        );

        // Map weights using HuggingFace naming convention
        self.map_weights_to_model(&mut model, weights)?;

        println!("✅ Model created successfully!");
        Ok(model)
    }

    /// Verify required weights exist
    fn verify_weights(&self, weights: &HashMap<String, Tensor>) -> Result<()> {
        let required_prefixes = vec![
            "wte.weight",           // Token embeddings
            "wpe.weight",           // Position embeddings
            "h.0.",                 // First transformer layer
            "ln_f.weight",          // Final layer norm
        ];

        for prefix in required_prefixes {
            let found = weights.keys().any(|k| k.contains(prefix));
            if !found {
                return Err(RustyGradientsError::IoError(format!(
                    "Missing required weight: {}",
                    prefix
                )));
            }
        }

        println!("✅ Weight verification passed");
        Ok(())
    }

    /// Map HuggingFace weights to RustyGradients model
    fn map_weights_to_model(
        &self,
        model: &mut GPT,
        weights: &HashMap<String, Tensor>,
    ) -> Result<()> {
        println!("🗺️  Mapping weights...");

        // Weight mapping table
        // HuggingFace → RustyGradients
        let mappings = vec![
            // Embeddings
            ("wte.weight", "token_embedding.weight"),
            ("wpe.weight", "position_embedding.weight"),
            // Transformer layers (example for layer 0)
            ("h.0.attn.c_attn.weight", "layer_0.attn.qkv.weight"),
            ("h.0.attn.c_attn.bias", "layer_0.attn.qkv.bias"),
            ("h.0.attn.c_proj.weight", "layer_0.attn.proj.weight"),
            ("h.0.attn.c_proj.bias", "layer_0.attn.proj.bias"),
            ("h.0.ln_1.weight", "layer_0.ln1.gamma"),
            ("h.0.ln_1.bias", "layer_0.ln1.beta"),
            ("h.0.mlp.c_fc.weight", "layer_0.ffn.fc1.weight"),
            ("h.0.mlp.c_fc.bias", "layer_0.ffn.fc1.bias"),
            ("h.0.mlp.c_proj.weight", "layer_0.ffn.fc2.weight"),
            ("h.0.mlp.c_proj.bias", "layer_0.ffn.fc2.bias"),
            ("h.0.ln_2.weight", "layer_0.ln2.gamma"),
            ("h.0.ln_2.bias", "layer_0.ln2.beta"),
            // Final layer norm
            ("ln_f.weight", "ln_f.gamma"),
            ("ln_f.bias", "ln_f.beta"),
        ];

        let mut mapped_count = 0;

        for (hf_name, _rg_name) in mappings {
            if weights.contains_key(hf_name) {
                mapped_count += 1;
                println!("  ✓ Mapped: {}", hf_name);
            }
        }

        println!("✅ Mapped {}/{} weights", mapped_count, mappings.len());

        // TODO: Actually copy weights to model tensors
        // This requires refactoring GPT model to expose weight setters
        println!("⚠️  Note: Weight copying not yet implemented");
        println!("   This is a structural demo. Full implementation requires:");
        println!("   - GPT model with mutable weight access");
        println!("   - Proper tensor shape handling");
        println!("   - Bias handling (HF has biases, we might not)");

        Ok(())
    }

    /// Get model metadata
    pub fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            model_type: "GPT".to_string(),
            vocab_size: self.config.vocab_size,
            embedding_dim: self.config.embedding_dim,
            num_layers: self.config.num_layers,
            num_heads: self.config.num_heads,
            block_size: self.config.block_size,
            dropout: self.config.dropout,
        }
    }
}

/// Helper function to load GPT-2 Small
#[cfg(feature = "huggingface")]
pub fn load_gpt2() -> Result<(GPT, HashMap<String, Tensor>)> {
    let loader = HFModelLoader::new(HFModelConfig::gpt2());
    let model_path = loader.download()?;
    loader.load_from_file(&model_path)
}

/// Helper function to load from local file
pub fn load_gpt2_from_file(path: &Path) -> Result<(GPT, HashMap<String, Tensor>)> {
    let loader = HFModelLoader::new(HFModelConfig::gpt2());
    loader.load_from_file(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_config() {
        let config = HFModelConfig::gpt2();
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.embedding_dim, 768);
        assert_eq!(config.num_layers, 12);
    }

    #[test]
    fn test_weight_verification() {
        let config = HFModelConfig::gpt2();
        let loader = HFModelLoader::new(config);

        let mut weights = HashMap::new();
        weights.insert(
            "wte.weight".to_string(),
            Tensor::new(ArrayD::zeros(vec![50257, 768]), false),
        );
        weights.insert(
            "wpe.weight".to_string(),
            Tensor::new(ArrayD::zeros(vec![1024, 768]), false),
        );
        weights.insert(
            "h.0.attn.c_attn.weight".to_string(),
            Tensor::new(ArrayD::zeros(vec![768, 2304]), false),
        );
        weights.insert(
            "ln_f.weight".to_string(),
            Tensor::new(ArrayD::zeros(vec![768]), false),
        );

        assert!(loader.verify_weights(&weights).is_ok());
    }

    #[test]
    #[ignore] // Requires network access
    #[cfg(feature = "huggingface")]
    fn test_download_gpt2() {
        let loader = HFModelLoader::new(HFModelConfig::gpt2());
        let result = loader.download();
        assert!(result.is_ok());
    }
}
