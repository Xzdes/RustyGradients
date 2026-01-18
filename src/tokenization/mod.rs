///! Tokenization module
///!
///! Provides multiple tokenization strategies:
///! - Character-level (simple, vocab=52)
///! - BPE (efficient, vocab=5,000+)
///! - HuggingFace tokenizers (GPT-2, LLaMA, etc.)

pub mod char_tokenizer;
pub mod bpe_tokenizer;

#[cfg(feature = "tokenization")]
pub mod hf_tokenizer;

pub use char_tokenizer::CharTokenizer;
pub use bpe_tokenizer::BPETokenizer;

#[cfg(feature = "tokenization")]
pub use hf_tokenizer::HFTokenizer;

use std::collections::HashMap;

/// Common tokenizer interface
pub trait Tokenizer: Send + Sync {
    /// Encode text to token IDs
    fn encode(&self, text: &str) -> Vec<usize>;

    /// Decode token IDs to text
    fn decode(&self, ids: &[usize]) -> String;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get special token IDs (PAD, BOS, EOS, UNK)
    fn special_tokens(&self) -> SpecialTokens;
}

/// Special token IDs
#[derive(Debug, Clone, Copy)]
pub struct SpecialTokens {
    pub pad: Option<usize>,
    pub bos: Option<usize>,
    pub eos: Option<usize>,
    pub unk: Option<usize>,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            pad: None,
            bos: None,
            eos: None,
            unk: None,
        }
    }
}
