///! Character-level tokenizer
///!
///! Simple tokenizer that maps each character to a unique token ID.
///! - Fast and simple
///! - Small vocabulary (typically 52-256 tokens)
///! - Good for learning and small datasets
///! - Not efficient for production use

use super::{SpecialTokens, Tokenizer};
use std::collections::HashMap;

/// Character-level tokenizer
#[derive(Clone, Debug)]
pub struct CharTokenizer {
    char_to_idx: HashMap<char, usize>,
    idx_to_char: HashMap<usize, char>,
    vocab_size: usize,
    special_tokens: SpecialTokens,
}

impl CharTokenizer {
    /// Create new character tokenizer from text corpus
    ///
    /// # Arguments
    /// * `text` - Training corpus to build vocabulary from
    ///
    /// # Example
    /// ```
    /// use rusty_gradients::tokenization::CharTokenizer;
    ///
    /// let text = "Hello, world!";
    /// let tokenizer = CharTokenizer::new(text);
    /// assert_eq!(tokenizer.vocab_size(), 10); // H, e, l, o, ',', ' ', w, r, d, !
    /// ```
    pub fn new(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let char_to_idx: HashMap<char, usize> = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        let idx_to_char: HashMap<usize, char> = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c))
            .collect();

        let vocab_size = chars.len();

        Self {
            char_to_idx,
            idx_to_char,
            vocab_size,
            special_tokens: SpecialTokens::default(),
        }
    }

    /// Create tokenizer with predefined vocabulary
    pub fn from_vocab(vocab: &[char]) -> Self {
        let char_to_idx: HashMap<char, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        let idx_to_char: HashMap<usize, char> = vocab
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c))
            .collect();

        Self {
            char_to_idx,
            idx_to_char,
            vocab_size: vocab.len(),
            special_tokens: SpecialTokens::default(),
        }
    }

    /// Get character for token ID
    pub fn get_char(&self, id: usize) -> Option<char> {
        self.idx_to_char.get(&id).copied()
    }

    /// Get token ID for character
    pub fn get_id(&self, c: char) -> Option<usize> {
        self.char_to_idx.get(&c).copied()
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.char_to_idx.get(&c).copied())
            .collect()
    }

    fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&i| self.idx_to_char.get(&i))
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn special_tokens(&self) -> SpecialTokens {
        self.special_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_tokenizer_basic() {
        let text = "Hello, world!";
        let tokenizer = CharTokenizer::new(text);

        // Test encoding
        let encoded = tokenizer.encode("Hello");
        assert_eq!(encoded.len(), 5);

        // Test decoding
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, "Hello");
    }

    #[test]
    fn test_char_tokenizer_vocab_size() {
        let text = "aabbcc";
        let tokenizer = CharTokenizer::new(text);
        assert_eq!(tokenizer.vocab_size(), 3); // a, b, c
    }

    #[test]
    fn test_char_tokenizer_from_vocab() {
        let vocab = vec!['a', 'b', 'c'];
        let tokenizer = CharTokenizer::from_vocab(&vocab);

        assert_eq!(tokenizer.vocab_size(), 3);
        assert_eq!(tokenizer.get_id('a'), Some(0));
        assert_eq!(tokenizer.get_id('b'), Some(1));
        assert_eq!(tokenizer.get_id('c'), Some(2));
    }

    #[test]
    fn test_char_tokenizer_roundtrip() {
        let text = "The quick brown fox jumps over the lazy dog.";
        let tokenizer = CharTokenizer::new(text);

        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, text);
    }
}
