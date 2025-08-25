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


// Ре-экспортируем самые важные структуры.
pub use module::Module;
pub use linear::Linear;
pub use activations::{ReLU, Sigmoid};
pub use containers::Sequential;
pub use embedding::Embedding;
pub use norm::LayerNorm;
pub use attention::MultiHeadAttention;
pub use feedforward::FeedForward;
pub use transformer::TransformerBlock;