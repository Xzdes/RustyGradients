// Объявляем файлы как публичные субмодули внутри `optim`.
pub mod optimizer;
pub mod sgd;

// -- Закомментируем пустой модуль-заглушку для будущего оптимизатора Adam --
// pub mod adam;


// Ре-экспортируем самые важные структуры для удобства.
pub use optimizer::Optimizer;
pub use sgd::SGD;