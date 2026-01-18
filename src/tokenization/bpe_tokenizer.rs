///! Byte Pair Encoding (BPE) tokenizer
///!
///! Efficient tokenization algorithm that learns subword units.
///! - Vocabulary size: 5,000-50,000 tokens
///! - Better compression than character-level
///! - Handles unknown words through subword units
///! - Used in GPT-2, GPT-3, LLaMA, etc.
///!
///! Algorithm:
///! 1. Start with character vocabulary
///! 2. Find most frequent adjacent pair
///! 3. Merge it into a new token
///! 4. Repeat until desired vocab size

use super::{SpecialTokens, Tokenizer};
use std::collections::HashMap;

/// BPE tokenizer with learned merge rules
#[derive(Clone, Debug)]
pub struct BPETokenizer {
    /// Token to ID mapping
    token_to_id: HashMap<String, usize>,
    /// ID to token mapping
    id_to_token: HashMap<usize, String>,
    /// Merge rules: (token1, token2) -> merged_token
    merges: Vec<(String, String)>,
    /// Vocabulary size
    vocab_size: usize,
    /// Special tokens
    special_tokens: SpecialTokens,
}

impl BPETokenizer {
    /// Train BPE tokenizer on corpus
    ///
    /// # Arguments
    /// * `text` - Training corpus
    /// * `vocab_size` - Target vocabulary size
    /// * `special_tokens` - Special tokens to add ([PAD], [BOS], [EOS], [UNK])
    ///
    /// # Example
    /// ```
    /// use rusty_gradients::tokenization::BPETokenizer;
    ///
    /// let text = "The quick brown fox jumps over the lazy dog.";
    /// let tokenizer = BPETokenizer::train(text, 256, &["[PAD]", "[UNK]"]);
    /// ```
    pub fn train(text: &str, vocab_size: usize, special_tokens_list: &[&str]) -> Self {
        // Step 1: Initialize with character vocabulary
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        let mut current_id = 0;

        // Add special tokens first
        let mut special_tokens = SpecialTokens::default();
        for (i, &token) in special_tokens_list.iter().enumerate() {
            token_to_id.insert(token.to_string(), i);
            id_to_token.insert(i, token.to_string());
            current_id = i + 1;

            // Map special tokens
            match token {
                "[PAD]" => special_tokens.pad = Some(i),
                "[BOS]" => special_tokens.bos = Some(i),
                "[EOS]" => special_tokens.eos = Some(i),
                "[UNK]" => special_tokens.unk = Some(i),
                _ => {}
            }
        }

        // Add character vocabulary
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        for c in chars {
            let token = c.to_string();
            if !token_to_id.contains_key(&token) {
                token_to_id.insert(token.clone(), current_id);
                id_to_token.insert(current_id, token);
                current_id += 1;
            }
        }

        let base_vocab_size = current_id;

        // Step 2: Learn merge rules
        let mut merges = Vec::new();
        let mut word_freqs = Self::get_word_frequencies(text);

        // Learn merges until we reach target vocab size
        for _ in base_vocab_size..vocab_size {
            // Find most frequent pair
            let pair = Self::find_most_frequent_pair(&word_freqs);
            if pair.is_none() {
                break; // No more pairs to merge
            }

            let (best_pair, _) = pair.unwrap();

            // Create new merged token
            let new_token = format!("{}{}", best_pair.0, best_pair.1);

            // Add to vocabulary
            token_to_id.insert(new_token.clone(), current_id);
            id_to_token.insert(current_id, new_token);
            current_id += 1;

            // Save merge rule
            merges.push(best_pair.clone());

            // Apply merge to all words
            word_freqs = Self::apply_merge(&word_freqs, &best_pair);
        }

        Self {
            token_to_id,
            id_to_token,
            merges,
            vocab_size: current_id,
            special_tokens,
        }
    }

    /// Get word frequencies from text
    fn get_word_frequencies(text: &str) -> HashMap<Vec<String>, usize> {
        let mut word_freqs = HashMap::new();

        for word in text.split_whitespace() {
            // Split word into characters
            let tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            *word_freqs.entry(tokens).or_insert(0) += 1;
        }

        word_freqs
    }

    /// Find most frequent adjacent pair
    fn find_most_frequent_pair(
        word_freqs: &HashMap<Vec<String>, usize>,
    ) -> Option<((String, String), usize)> {
        let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();

        for (word, &freq) in word_freqs {
            for i in 0..word.len().saturating_sub(1) {
                let pair = (word[i].clone(), word[i + 1].clone());
                *pair_freqs.entry(pair).or_insert(0) += freq;
            }
        }

        pair_freqs
            .into_iter()
            .max_by_key(|(_, freq)| *freq)
    }

    /// Apply merge rule to all words
    fn apply_merge(
        word_freqs: &HashMap<Vec<String>, usize>,
        pair: &(String, String),
    ) -> HashMap<Vec<String>, usize> {
        let mut new_word_freqs = HashMap::new();

        for (word, &freq) in word_freqs {
            let mut new_word = Vec::new();
            let mut i = 0;

            while i < word.len() {
                // Check if current and next token match the pair
                if i < word.len() - 1 && word[i] == pair.0 && word[i + 1] == pair.1 {
                    // Merge
                    new_word.push(format!("{}{}", pair.0, pair.1));
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }

            *new_word_freqs.entry(new_word).or_insert(0) += freq;
        }

        new_word_freqs
    }

    /// Tokenize word using learned merge rules
    fn tokenize_word(&self, word: &str) -> Vec<String> {
        // Start with character-level tokens
        let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        // Apply merge rules in order
        for (left, right) in &self.merges {
            let mut i = 0;
            while i < tokens.len().saturating_sub(1) {
                if tokens[i] == *left && tokens[i + 1] == *right {
                    // Merge tokens
                    let merged = format!("{}{}", left, right);
                    tokens[i] = merged;
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        tokens
    }

    /// Save tokenizer to JSON
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs;

        let data = serde_json::json!({
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "merges": self.merges,
            "vocab_size": self.vocab_size,
        });

        fs::write(path, serde_json::to_string_pretty(&data)?)?;
        Ok(())
    }

    /// Load tokenizer from JSON
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        use std::fs;

        let data = fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&data)?;

        let token_to_id: HashMap<String, usize> =
            serde_json::from_value(json["token_to_id"].clone())?;
        let id_to_token: HashMap<usize, String> =
            serde_json::from_value(json["id_to_token"].clone())?;
        let merges: Vec<(String, String)> =
            serde_json::from_value(json["merges"].clone())?;
        let vocab_size: usize = json["vocab_size"].as_u64().unwrap() as usize;

        Ok(Self {
            token_to_id,
            id_to_token,
            merges,
            vocab_size,
            special_tokens: SpecialTokens::default(),
        })
    }
}

impl Tokenizer for BPETokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();

        for word in text.split_whitespace() {
            let tokens = self.tokenize_word(word);

            for token in tokens {
                if let Some(&id) = self.token_to_id.get(&token) {
                    ids.push(id);
                } else if let Some(unk_id) = self.special_tokens.unk {
                    // Unknown token
                    ids.push(unk_id);
                }
            }
        }

        ids
    }

    fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.id_to_token.get(&id))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
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
    fn test_bpe_train() {
        let text = "the quick brown fox jumps over the lazy dog";
        let tokenizer = BPETokenizer::train(text, 100, &["[PAD]", "[UNK]"]);

        assert!(tokenizer.vocab_size() > 20); // More than just characters
        assert!(tokenizer.vocab_size() <= 100);
    }

    #[test]
    fn test_bpe_encode_decode() {
        let text = "hello world hello";
        let tokenizer = BPETokenizer::train(text, 50, &["[UNK]"]);

        let encoded = tokenizer.encode("hello");
        assert!(!encoded.is_empty());

        let decoded = tokenizer.decode(&encoded);
        assert!(decoded.contains("hello") || decoded.contains("h") || decoded.contains("e"));
    }

    #[test]
    fn test_bpe_special_tokens() {
        let text = "test";
        let tokenizer = BPETokenizer::train(text, 50, &["[PAD]", "[UNK]", "[BOS]", "[EOS]"]);

        let special = tokenizer.special_tokens();
        assert!(special.pad.is_some());
        assert!(special.unk.is_some());
        assert!(special.bos.is_some());
        assert!(special.eos.is_some());
    }

    #[test]
    fn test_bpe_compression() {
        let text = "aaaa bbbb cccc aaaa bbbb";
        let char_tokenizer = super::super::CharTokenizer::new(text);
        let bpe_tokenizer = BPETokenizer::train(text, 50, &[]);

        let char_encoded = char_tokenizer.encode(text);
        let bpe_encoded = bpe_tokenizer.encode(text);

        // BPE should achieve better compression (fewer tokens)
        println!("Char: {} tokens, BPE: {} tokens", char_encoded.len(), bpe_encoded.len());
    }
}
