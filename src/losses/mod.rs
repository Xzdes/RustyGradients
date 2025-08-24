// Объявляем файл `mse.rs` как публичный субмодуль.
pub mod mse;

// Ре-экспортируем функцию `mse_loss` для удобства,
// чтобы можно было писать `use slmrustai::losses::mse_loss;`
pub use mse::mse_loss;