//! Модуль, реализующий функцию потерь Среднеквадратичной ошибки (MSE).

use crate::tensor::Tensor;

/// Вычисляет Среднеквадратичную ошибку (Mean Squared Error).
///
/// Эта функция потерь часто используется для задач регрессии.
///
/// Формула: `MSE = mean((y_pred - y_true)^2)`.
/// В данной реализации для простоты вычисляется сумма, а не среднее:
/// `sum((y_pred - y_true)^2)`. Для процесса оптимизации это эквивалентно,
/// так как константный множитель (1/N) влияет только на абсолютное значение
/// потерь, но не на направление градиента.
///
/// # Аргументы
///
/// * `y_pred` - Тензор с предсказаниями модели.
/// * `y_true` - Тензор с истинными значениями.
///
/// # Возвращает
///
/// Скалярный `Tensor` (0-мерный), содержащий значение ошибки.
///
/// # Примеры
///
/// ```
/// # use rusty_gradients::tensor::Tensor;
/// # use rusty_gradients::losses::mse_loss;
/// # use ndarray::array;
/// let predictions = Tensor::new(array![0.9, 0.1, 0.6].into_dyn(), true);
/// let targets = Tensor::new(array![1.0, 0.0, 0.5].into_dyn(), false);
///
/// let loss = mse_loss(&predictions, &targets);
///
/// // loss = (0.9-1.0)^2 + (0.1-0.0)^2 + (0.6-0.5)^2
/// // loss = (-0.1)^2 + (0.1)^2 + (0.1)^2
/// // loss = 0.01 + 0.01 + 0.01 = 0.03
/// assert!((loss.data.borrow().sum() - 0.03).abs() < 1e-6);
/// ```
pub fn mse_loss(y_pred: &Tensor, y_true: &Tensor) -> Tensor {
    let error = y_pred.sub(y_true);
    let loss = error.powf(2.0).sum();
    loss
}