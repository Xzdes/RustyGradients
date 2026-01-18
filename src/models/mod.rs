// src/models/mod.rs

// Объявляем, что внутри нашего модуля `models` есть подмодуль `gpt`.
// Это заставит Rust искать файл `src/models/gpt.rs`.
pub mod gpt;

// HuggingFace model loader (requires serialization feature)
#[cfg(any(feature = "serialization", feature = "huggingface"))]
pub mod hf_loader;