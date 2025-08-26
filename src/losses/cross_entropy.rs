// src/losses/cross_entropy.rs
use crate::tensor::Tensor;

/// Вычисляет Cross-Entropy Loss, вызывая стабильную и графо-безопасную операцию
/// `sparse_cross_entropy_op` из `Tensor`.
///
/// # Аргументы
///
/// * `logits` - сырые выходы модели.
/// * `targets` - тензор с правильными индексами.
///
/// # Возвращает
///
/// Скалярный тензор со значением ошибки.
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    logits.sparse_cross_entropy(targets)
}