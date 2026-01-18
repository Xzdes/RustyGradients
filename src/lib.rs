// src/lib.rs
//! # RustyGradients
//!
//! `rusty-gradients` - это фреймворк для глубокого обучения на Rust,
//! вдохновленный PyTorch, с фокусом на простоте, производительности и
//! реализации современных архитектур, таких как Трансформеры.
//!
//! Основной компонент - это `Tensor`, который поддерживает автоматическое
//! дифференцирование (autograd).

pub mod error;
pub mod tensor;
pub mod core;
pub mod nn;
pub mod ops;
pub mod optim;
pub mod losses;

// --- ИСПРАВЛЕНИЕ: Объявляем наш новый модуль с моделью ---
pub mod models;

// --- Объявляем наш API для WebAssembly ---
pub mod wasm_api;

// === NEW: Backend abstraction layer ===
pub mod backend;

// === NEW: TensorV2 with multi-backend support ===
pub mod tensor_v2;