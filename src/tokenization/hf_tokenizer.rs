///! HuggingFace tokenizers integration
///!
///! Allows loading pre-trained tokenizers from HuggingFace:
///! - GPT-2 tokenizer (50,257 vocab)
///! - LLaMA tokenizer (32,000 vocab)
///! - BERT tokenizer (30,522 vocab)
///! - And more...
///!
///! Requires `tokenization` feature flag.

use super::{SpecialTokens, Tokenizer};
use tokenizers::Tokenizer as HFTokenizerImpl;
use crate::error::{Result, RustyGradientsError};

/// HuggingFace tokenizer wrapper
pub struct HFTokenizer {
    tokenizer: HFTokenizerImpl,
    special_tokens: SpecialTokens,
}

impl HFTokenizer {
    /// Load tokenizer from HuggingFace Hub
    ///
    /// # Arguments
    /// * `model_name` - Model identifier (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
    ///
    /// # Example
    /// ```no_run
    /// use rusty_gradients::tokenization::HFTokenizer;
    ///
    /// let tokenizer = HFTokenizer::from_pretrained("gpt2").unwrap();
    /// let encoded = tokenizer.encode("Hello, world!");
    /// ```
    /// Load tokenizer from HuggingFace Hub (requires network access)
    ///
    /// Note: This requires the tokenizer to be already downloaded.
    /// Use `from_file` with a local tokenizer.json for offline usage.
    #[cfg(feature = "huggingface")]
    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        // Use hf-hub to download tokenizer.json
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| RustyGradientsError::IoError(format!("Failed to init HF API: {}", e)))?;
        let repo = api.model(model_name.to_string());
        let tokenizer_path = repo.get("tokenizer.json")
            .map_err(|e| RustyGradientsError::IoError(format!("Failed to download tokenizer: {}", e)))?;

        Self::from_file(tokenizer_path.to_str().unwrap_or("tokenizer.json"))
    }

    /// Stub for when huggingface feature is not enabled
    #[cfg(not(feature = "huggingface"))]
    pub fn from_pretrained(_model_name: &str) -> Result<Self> {
        Err(RustyGradientsError::IoError(
            "from_pretrained requires 'huggingface' feature. Use from_file() instead.".to_string()
        ))
    }

    /// Load tokenizer from file
    ///
    /// # Arguments
    /// * `path` - Path to tokenizer.json file
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = HFTokenizerImpl::from_file(path)
            .map_err(|e| RustyGradientsError::IoError(format!("Failed to load tokenizer file: {}", e)))?;
        let special_tokens = Self::extract_special_tokens(&tokenizer);

        Ok(Self {
            tokenizer,
            special_tokens,
        })
    }

    /// Extract special token IDs from tokenizer
    fn extract_special_tokens(tokenizer: &HFTokenizerImpl) -> SpecialTokens {
        let vocab = tokenizer.get_vocab(true);

        SpecialTokens {
            pad: vocab.get("<|pad|>").or_else(|| vocab.get("[PAD]")).map(|&v| v as usize),
            bos: vocab
                .get("<|startoftext|>")
                .or_else(|| vocab.get("[BOS]"))
                .or_else(|| vocab.get("<s>"))
                .map(|&v| v as usize),
            eos: vocab
                .get("<|endoftext|>")
                .or_else(|| vocab.get("[EOS]"))
                .or_else(|| vocab.get("</s>"))
                .map(|&v| v as usize),
            unk: vocab.get("[UNK]").or_else(|| vocab.get("<unk>")).map(|&v| v as usize),
        }
    }

    /// Encode with options
    pub fn encode_with_options(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<usize>> {
        let encoding = self.tokenizer.encode(text, add_special_tokens)
            .map_err(|e| RustyGradientsError::InvalidInput(format!("Encoding failed: {}", e)))?;
        Ok(encoding.get_ids().iter().map(|&id| id as usize).collect())
    }

    /// Batch encode multiple texts
    pub fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<Vec<Vec<usize>>> {
        let encodings = self.tokenizer.encode_batch(texts.to_vec(), add_special_tokens)
            .map_err(|e| RustyGradientsError::InvalidInput(format!("Batch encoding failed: {}", e)))?;

        Ok(encodings
            .iter()
            .map(|enc| enc.get_ids().iter().map(|&id| id as usize).collect())
            .collect())
    }

    /// Get the underlying HuggingFace tokenizer
    pub fn inner(&self) -> &HFTokenizerImpl {
        &self.tokenizer
    }
}

impl Tokenizer for HFTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        self.encode_with_options(text, true)
            .unwrap_or_else(|_| Vec::new())
    }

    fn decode(&self, ids: &[usize]) -> String {
        let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.tokenizer
            .decode(&ids_u32, true)
            .unwrap_or_else(|_| String::new())
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    fn special_tokens(&self) -> SpecialTokens {
        self.special_tokens
    }
}

/// Helper function to download and cache GPT-2 tokenizer
pub fn load_gpt2_tokenizer() -> Result<HFTokenizer> {
    HFTokenizer::from_pretrained("gpt2")
}

/// Helper function to download and cache LLaMA tokenizer
pub fn load_llama_tokenizer() -> Result<HFTokenizer> {
    HFTokenizer::from_pretrained("meta-llama/Llama-2-7b-hf")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires network access
    fn test_load_gpt2() {
        let tokenizer = load_gpt2_tokenizer().unwrap();
        assert_eq!(tokenizer.vocab_size(), 50257);

        let text = "Hello, world!";
        let encoded = tokenizer.encode(text);
        assert!(!encoded.is_empty());

        let decoded = tokenizer.decode(&encoded);
        assert!(decoded.contains("Hello"));
    }

    #[test]
    #[ignore] // Requires network access
    fn test_encode_batch() {
        let tokenizer = load_gpt2_tokenizer().unwrap();

        let texts = vec!["Hello", "World", "Test"];
        let batch = tokenizer.encode_batch(&texts, true).unwrap();

        assert_eq!(batch.len(), 3);
        for encoded in batch {
            assert!(!encoded.is_empty());
        }
    }
}
