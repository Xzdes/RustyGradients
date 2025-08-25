//! Модуль, содержащий трейт `Optimizer` и его реализации.
//!
//! Этот модуль предоставляет алгоритмы оптимизации, используемые для обновления
//! параметров модели в процессе обучения.
//!
//! # Основные компоненты
//!
//! - [`Optimizer`](optimizer::Optimizer): Основной трейт, который должны реализовывать все оптимизаторы.
//! - [`SGD`](sgd::SGD): Реализация стохастического градиентного спуска.
//! - [`Adam`](adam::Adam): Реализация популярного адаптивного оптимизатора Adam.
//!
//! # Пример использования
//!
//! # use rusty_gradients::nn::{Linear, Module};
//! # use rusty_gradients::optim::{Optimizer, SGD};
//! # use rusty_gradients::tensor::Tensor;
//!
//! // 1. Создаем модель
//! let model = Linear::new(10, 1);
//!
//! // 2. Создаем оптимизатор, передавая ему параметры модели и скорость обучения
//! let mut optimizer = SGD::new(model.parameters(), 0.01);
//!
//! // ... внутри цикла обучения ...
//! # // --- ИСПРАВЛЕНИЕ: Указываем корректную форму для тензора ---
//! # let dummy_input = Tensor::zeros(&, false);
//! # let loss = model.forward(&dummy_input).unwrap();
//! // loss.backward();
//! // optimizer.step();
//! // optimizer.zero_grad();
//! ```

// Объявляем файлы как публичные субмодули внутри `optim`.
pub mod optimizer;
pub mod sgd;
pub mod adam;


// Ре-экспортируем самые важные структуры для удобства.
pub use optimizer::Optimizer;
pub use sgd::SGD;
pub use adam::Adam;