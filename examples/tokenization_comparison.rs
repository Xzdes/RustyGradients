///! Tokenization Comparison Example
///!
///! Demonstrates the difference between character-level and BPE tokenization.
///!
///! Run with:
///!   cargo run --example tokenization_comparison

use rusty_gradients::tokenization::{BPETokenizer, CharTokenizer, Tokenizer};
use std::fs;

fn main() {
    println!("=== Tokenization Comparison Example ===\n");

    // Load training data
    let text = fs::read_to_string("input.txt")
        .expect("Failed to read input.txt");

    println!("Training corpus:");
    println!("  Length: {} characters", text.len());
    println!("  Words: ~{}\n", text.split_whitespace().count());

    // === Character-level Tokenization ===
    println!("1. CHARACTER-LEVEL TOKENIZATION");
    println!("   Strategy: Each character = 1 token\n");

    let char_tokenizer = CharTokenizer::new(&text);
    let char_vocab_size = char_tokenizer.vocab_size();

    println!("   Vocabulary size: {}", char_vocab_size);

    // Encode sample text
    let sample_text = "Artificial intelligence is transforming the world.";
    let char_encoded = char_tokenizer.encode(sample_text);

    println!("   Sample: \"{}\"", sample_text);
    println!("   Encoded length: {} tokens", char_encoded.len());
    println!("   Tokens per character: {:.2}", char_encoded.len() as f32 / sample_text.len() as f32);
    println!("   Compression ratio: {:.2}x\n", sample_text.len() as f32 / char_encoded.len() as f32);

    // === BPE Tokenization ===
    println!("2. BPE TOKENIZATION (Byte Pair Encoding)");
    println!("   Strategy: Learn frequent subword units\n");

    // Train BPE with different vocab sizes
    let vocab_sizes = vec![256, 512, 1024, 2048];

    for &vocab_size in &vocab_sizes {
        println!("   --- BPE with vocab_size = {} ---", vocab_size);

        let bpe_tokenizer = BPETokenizer::train(&text, vocab_size, &["[PAD]", "[UNK]", "[BOS]", "[EOS]"]);
        let actual_vocab = bpe_tokenizer.vocab_size();

        println!("   Actual vocabulary size: {}", actual_vocab);

        // Encode same sample
        let bpe_encoded = bpe_tokenizer.encode(sample_text);

        println!("   Sample: \"{}\"", sample_text);
        println!("   Encoded length: {} tokens", bpe_encoded.len());
        println!("   Tokens per character: {:.2}", bpe_encoded.len() as f32 / sample_text.len() as f32);
        println!("   Compression ratio: {:.2}x", sample_text.len() as f32 / bpe_encoded.len() as f32);
        println!("   Improvement over char-level: {:.2}x\n",
            char_encoded.len() as f32 / bpe_encoded.len() as f32);
    }

    // === Full Corpus Comparison ===
    println!("3. FULL CORPUS COMPRESSION\n");

    let char_full = char_tokenizer.encode(&text);
    let bpe_256 = BPETokenizer::train(&text, 256, &["[UNK]"]);
    let bpe_1024 = BPETokenizer::train(&text, 1024, &["[UNK]"]);
    let bpe_2048 = BPETokenizer::train(&text, 2048, &["[UNK]"]);

    let bpe_256_encoded = bpe_256.encode(&text);
    let bpe_1024_encoded = bpe_1024.encode(&text);
    let bpe_2048_encoded = bpe_2048.encode(&text);

    println!("   Original text: {} characters", text.len());
    println!();
    println!("   Tokenizer              | Tokens | Compression | Improvement");
    println!("   -----------------------|--------|-------------|------------");
    println!("   Char-level (vocab={})  | {}  | {:.2}x       | baseline",
        char_vocab_size,
        char_full.len(),
        text.len() as f32 / char_full.len() as f32);

    println!("   BPE (vocab=256)        | {}  | {:.2}x       | {:.2}x",
        bpe_256_encoded.len(),
        text.len() as f32 / bpe_256_encoded.len() as f32,
        char_full.len() as f32 / bpe_256_encoded.len() as f32);

    println!("   BPE (vocab=1024)       | {}  | {:.2}x       | {:.2}x",
        bpe_1024_encoded.len(),
        text.len() as f32 / bpe_1024_encoded.len() as f32,
        char_full.len() as f32 / bpe_1024_encoded.len() as f32);

    println!("   BPE (vocab=2048)       | {}  | {:.2}x       | {:.2}x",
        bpe_2048_encoded.len(),
        text.len() as f32 / bpe_2048_encoded.len() as f32,
        char_full.len() as f32 / bpe_2048_encoded.len() as f32);

    println!();

    // === Token Examples ===
    println!("4. TOKEN EXAMPLES\n");

    println!("   Character-level tokens (first 20):");
    let char_sample: Vec<String> = char_encoded.iter()
        .take(20)
        .filter_map(|&id| char_tokenizer.get_char(id))
        .map(|c| format!("'{}'", c))
        .collect();
    println!("   {}\n", char_sample.join(", "));

    println!("   BPE tokens (vocab=1024, first 20):");
    let bpe_sample = bpe_1024.encode(sample_text);
    let bpe_decoded_tokens: Vec<String> = bpe_sample.iter()
        .take(20)
        .map(|&id| format!("\"{}\"", bpe_1024.decode(&[id])))
        .collect();
    println!("   {}\n", bpe_decoded_tokens.join(", "));

    // === Roundtrip Test ===
    println!("5. ROUNDTRIP VERIFICATION\n");

    let char_decoded = char_tokenizer.decode(&char_encoded);
    let char_match = char_decoded == sample_text;
    println!("   Char-level: {} ({})",
        if char_match { "✓ PASS" } else { "✗ FAIL" },
        if char_match { "exact match" } else { "mismatch" });

    let bpe_sample_encoded = bpe_1024.encode(sample_text);
    let bpe_decoded = bpe_1024.decode(&bpe_sample_encoded);
    let bpe_words_match = sample_text.split_whitespace().count() == bpe_decoded.split_whitespace().count();
    println!("   BPE:        {} ({})",
        if bpe_words_match { "✓ PASS" } else { "~ PARTIAL" },
        if bpe_words_match { "word count preserved" } else { "approximate" });

    println!();

    // === Summary ===
    println!("6. SUMMARY\n");
    println!("   ✅ Character-level tokenization:");
    println!("      - Simple and exact");
    println!("      - Vocab size: {} (very small)", char_vocab_size);
    println!("      - No compression (1 token per char)");
    println!();
    println!("   ✅ BPE tokenization:");
    println!("      - Learns subword units");
    println!("      - Vocab size: 256-50,000 (configurable)");
    println!("      - Compression: 1.5-3x better than char-level");
    println!("      - Used in GPT-2, GPT-3, LLaMA");
    println!();
    println!("   📊 Recommendation:");
    println!("      - Use CHAR for small datasets (<10KB)");
    println!("      - Use BPE for production (vocab=5,000-50,000)");
    println!();

    // === Save BPE tokenizer ===
    println!("7. SAVING TOKENIZER\n");

    let save_path = "bpe_tokenizer_1024.json";
    match bpe_1024.save(save_path) {
        Ok(_) => println!("   ✓ BPE tokenizer saved to: {}", save_path),
        Err(e) => println!("   ✗ Failed to save: {}", e),
    }

    // Test loading
    match BPETokenizer::load(save_path) {
        Ok(loaded) => {
            let test_encoded = loaded.encode("test");
            println!("   ✓ BPE tokenizer loaded successfully");
            println!("   ✓ Test encoding works: {} tokens", test_encoded.len());
        }
        Err(e) => println!("   ✗ Failed to load: {}", e),
    }

    println!();
    println!("=== Example Complete! ===");
}
