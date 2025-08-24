use crate::tensor::Tensor;

/// Вычисляет Среднеквадратичную ошибку (Mean Squared Error).
///
/// Формула: `MSE = mean((y_pred - y_true)^2)`.
/// Мы для простоты реализуем `sum((y_pred - y_true)^2)`,
/// так как для градиента константный множитель (1/N) не имеет значения.
///
/// # Аргументы
///
/// * `y_pred` - Тензор с предсказаниями модели.
/// * `y_true` - Тензор с истинными значениями.
///
/// # Возвращает
///
/// Скалярный `Tensor` (0-мерный), содержащий значение ошибки.
pub fn mse_loss(y_pred: &Tensor, y_true: &Tensor) -> Tensor {
    let error = y_pred.sub(y_true);
    let loss = error.powf(2.0).sum();
    loss
}