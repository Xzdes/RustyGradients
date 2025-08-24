// Объявляем файлы как публичные субмодули внутри `nn`.
pub mod linear;
pub mod module;

// -- Закомментируем пустые модули-заглушки для будущей работы --
// pub mod activations;
// pub mod containers;
// pub mod embedding;
// pub mod norm;
// pub mod attention;


// Ре-экспортируем самые важные структуры, чтобы можно было писать
// `use slmrustai::nn::{Module, Linear};` вместо длинных путей.
pub use module::Module;
pub use linear::Linear;