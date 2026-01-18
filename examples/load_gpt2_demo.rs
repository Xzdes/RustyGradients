///! Load GPT-2 from HuggingFace Demo
///!
///! Demonstrates loading a pre-trained GPT-2 model from HuggingFace Hub.
///!
///! Run with:
///!   cargo run --example load_gpt2_demo --features "serialization huggingface"
///!
///! Note: Requires internet connection for first run (downloads model)

#[cfg(all(feature = "serialization", feature = "huggingface"))]
use rusty_gradients::models::hf_loader::{HFModelConfig, HFModelLoader};

fn main() {
    #[cfg(not(all(feature = "serialization", feature = "huggingface")))]
    {
        println!("❌ This example requires both 'serialization' and 'huggingface' features");
        println!();
        println!("Run with:");
        println!("  cargo run --example load_gpt2_demo --features \"serialization huggingface\"");
        std::process::exit(1);
    }

    #[cfg(all(feature = "serialization", feature = "huggingface"))]
    {
        println!("=== HuggingFace GPT-2 Loader Demo ===\n");

        // Available GPT-2 models
        println!("📦 Available GPT-2 Models:\n");
        println!("  1. GPT-2 Small  (124M params)  - Default");
        println!("  2. GPT-2 Medium (355M params)");
        println!("  3. GPT-2 Large  (774M params)");
        println!("  4. GPT-2 XL     (1.5B params)");
        println!();

        // Load GPT-2 Small
        println!("🔽 Loading GPT-2 Small (124M parameters)...\n");

        let config = HFModelConfig::gpt2();
        println!("Configuration:");
        println!("  Model: {}", config.model_name);
        println!("  Vocabulary: {} tokens", config.vocab_size);
        println!("  Embedding dim: {}", config.embedding_dim);
        println!("  Layers: {}", config.num_layers);
        println!("  Heads: {}", config.num_heads);
        println!("  Context window: {} tokens", config.block_size);
        println!();

        let loader = HFModelLoader::new(config);

        println!("📥 Downloading model from HuggingFace Hub...");
        println!("   (First run only, subsequent runs use cache)");
        println!();

        match loader.download() {
            Ok(model_path) => {
                println!("✅ Model downloaded successfully!");
                println!("   Path: {}", model_path.display());
                println!();

                println!("📂 Loading weights...");
                match loader.load_from_file(&model_path) {
                    Ok((_model, weights)) => {
                        println!("✅ Model loaded successfully!");
                        println!();
                        println!("📊 Model Statistics:");
                        println!("   Total weights: {}", weights.len());
                        println!();

                        // Show some weight info
                        println!("🔍 Sample Weights:");
                        for (name, tensor) in weights.iter().take(5) {
                            let shape = tensor.data.borrow().shape().to_vec();
                            let total_params: usize = shape.iter().product();
                            println!("   - {}: {:?} ({} params)",
                                name,
                                shape,
                                format_number(total_params));
                        }

                        // Calculate total parameters
                        let total_params: usize = weights
                            .values()
                            .map(|t| t.data.borrow().shape().iter().product::<usize>())
                            .sum();

                        println!();
                        println!("📊 Total Parameters: {}", format_number(total_params));

                        // Memory usage estimate
                        let memory_mb = (total_params * 4) as f64 / 1024.0 / 1024.0; // 4 bytes per f32
                        println!("💾 Memory Usage: {:.1} MB (fp32)", memory_mb);
                        println!();

                        println!("🎉 Demo Complete!");
                        println!();
                        println!("Next steps:");
                        println!("  ✓ Model is loaded and ready");
                        println!("  ⏳ Integration with inference pipeline (coming soon)");
                        println!("  ⏳ Fine-tuning support (coming soon)");
                    }
                    Err(e) => {
                        println!("❌ Failed to load model: {}", e);
                        println!();
                        println!("This is expected! Weight mapping is not yet fully implemented.");
                        println!("The model file was downloaded successfully, but we need to:");
                        println!("  1. Implement full weight mapping (HF → RustyGradients)");
                        println!("  2. Handle bias terms");
                        println!("  3. Implement weight setter methods in GPT model");
                    }
                }
            }
            Err(e) => {
                println!("❌ Failed to download model: {}", e);
                println!();
                println!("Possible causes:");
                println!("  - No internet connection");
                println!("  - HuggingFace Hub is down");
                println!("  - Authentication required (for some models)");
                println!();
                println!("Try:");
                println!("  - Check your internet connection");
                println!("  - Visit https://huggingface.co/gpt2 to verify model exists");
            }
        }

        println!();
        println!("=== Demo Complete ===");
    }
}

/// Format large numbers with commas
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
