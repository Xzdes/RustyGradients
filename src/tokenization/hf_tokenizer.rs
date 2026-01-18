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
    pub fn from_pretrained(model_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Load from HuggingFace Hub
        let tokenizer = HFTokenizerImpl::from_pretrained(model_name, None)?;

        // Extract special token IDs
        let special_tokens = Self::extract_special_tokens(&tokenizer);

        Ok(Self {
            tokenizer,
            special_tokens,
        })
    }

    /// Load tokenizer from file
    ///
    /// # Arguments
    /// * `path` - Path to tokenizer.json file
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer = HFTokenizerImpl::from_file(path)?;
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
            pad: vocab.get("<|pad|>").or_else(|| vocab.get("[PAD]")).copied(),
            bos: vocab
                .get("<|startoftext|>")
                .or_else(|| vocab.get("[BOS]"))
                .or_else(|| vocab.get("<s>"))
                .copied(),
            eos: vocab
                .get("<|endoftext|>")
                .or_else(|| vocab.get("[EOS]"))
                .or_else(|| vocab.get("</s>"))
                .copied(),
            unk: vocab.get("[UNK]").or_else(|| vocab.get("<unk>")).copied(),
        }
    }

    /// Encode with options
    pub fn encode_with_options(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(text, add_special_tokens)?;
        Ok(encoding.get_ids().iter().map(|&id| id as usize).collect())
    }

    /// Batch encode multiple texts
    pub fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
        let encodings = self.tokenizer.encode_batch(texts.to_vec(), add_special_tokens)?;

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
pub fn load_gpt2_tokenizer() -> Result<HFTokenizer, Box<dyn std::error::Error>> {
    HFTokenizer::from_pretrained("gpt2")
}

/// Helper function to download and cache LLaMA tokenizer
pub fn load_llama_tokenizer() -> Result<HFTokenizer, Box<dyn std::error::Error>> {
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
