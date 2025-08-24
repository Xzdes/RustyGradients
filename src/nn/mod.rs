// Объявляем файлы как публичные субмодули внутри `nn`.
pub mod linear;
pub mod module;
pub mod activations;
pub mod containers;

// -- Оставляем эти модули закомментированными, так как мы их еще не создали --
// pub mod embedding;
// pub mod norm;
// pub mod attention;


// Ре-экспортируем самые важные структуры, чтобы можно было писать
// `use slmrustai::nn::{Module, Linear, ReLU, Sequential, Sigmoid};` вместо длинных путей.
pub use module::Module;
pub use linear::Linear;
pub use activations::{ReLU, Sigmoid}; // <-- Добавляем Sigmoid сюда
pub use containers::Sequential;