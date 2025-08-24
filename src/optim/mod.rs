// Объявляем файлы как публичные субмодули внутри `optim`.
pub mod optimizer;
pub mod sgd;
pub mod adam; // <-- Раскомментируем (или добавляем) эту строку


// Ре-экспортируем самые важные структуры для удобства.
pub use optimizer::Optimizer;
pub use sgd::SGD;
pub use adam::Adam; // <-- Добавляем Adam в публичный экспорт