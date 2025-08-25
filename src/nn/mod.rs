//! Модуль, содержащий строительные блоки для нейронных сетей.
//!
//! Этот модуль является аналогом `torch.nn` в PyTorch. Он предоставляет
//! набор готовых слоев, контейнеров и утилит для легкого построения
//! сложных нейросетевых архитектур.
//!
//! # Основные компоненты
//!
//! - [`Module`](module::Module): Основной трейт, который должны реализовывать все слои.
//! - [`Linear`](linear::Linear): Полносвязный слой.
//! - [`ReLU`](activations::ReLU), [`Sigmoid`](activations::Sigmoid): Слои активации.
//! - [`Embedding`](embedding::Embedding): Слой для представления дискретных данных.
//! - [`LayerNorm`](norm::LayerNorm): Слой нормализации.
//! - [`MultiHeadAttention`](attention::MultiHeadAttention): Ключевой компонент Трансформеров.
//! - [`TransformerBlock`](transformer::TransformerBlock): Готовый блок кодировщика Трансформера.
//! - [`Sequential`](containers::Sequential): Контейнер для последовательного объединения слоев.

// Объявляем файлы как публичные субмодули внутри `nn`.
pub mod linear;
pub mod module;
pub mod activations;
pub mod containers;
pub mod embedding;
pub mod norm;
pub mod attention;
pub mod feedforward;
pub mod transformer;


// Ре-экспортируем самые важные структуры для удобства.
pub use module::Module;
pub use linear::Linear;
pub use activations::{ReLU, Sigmoid};
pub use containers::Sequential;
pub use embedding::Embedding;
pub use norm::LayerNorm;
pub use attention::MultiHeadAttention;
pub use feedforward::FeedForward;
pub use transformer::TransformerBlock;