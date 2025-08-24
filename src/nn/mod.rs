// Объявляем файлы как публичные субмодули внутри `nn`.
pub mod linear;
pub mod module;
pub mod activations;
pub mod containers;
pub mod embedding; // <-- Раскомментируем эту строку

// -- Оставляем эти модули закомментированными --
// pub mod norm;
// pub mod attention;


// Ре-экспортируем самые важные структуры.
pub use module::Module;
pub use linear::Linear;
pub use activations::{ReLU, Sigmoid};
pub use containers::Sequential;
pub use embedding::Embedding; // <-- Добавляем Embedding в публичный экспорт