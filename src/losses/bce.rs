//! Модуль, реализующий функцию потерь Бинарной Перекрестной Энтропии (BCE).

use crate::tensor::Tensor;
use std::ops::{Add, Mul};

/// Вычисляет Бинарную Перекрестную Энтропию (Binary Cross-Entropy Loss).
///
/// Эта функция потерь идеально подходит для задач бинарной классификации (когда
/// есть только два класса, 0 и 1). Она ожидает, что предсказания модели
/// (`p`) находятся в диапазоне `[0, 1]`, что обычно достигается с помощью
/// `sigmoid` активации на выходе.
///
/// Формула: `Loss = -sum(y * log(p) + (1 - y) * log(1 - p))`.
///
/// # Аргументы
///
/// * `p` - Тензор с предсказанными вероятностями (выходы модели после `sigmoid`).
/// * `y` - Тензор с истинными метками (должны быть 0.0 или 1.0).
///
/// # Возвращает
///
/// Скалярный `Tensor`, содержащий значение ошибки.
///
/// # Примеры
///
/// # use rusty_gradients::tensor::Tensor;
/// # use rusty_gradients::losses::bce_loss;
/// # use ndarray::array;
/// let predictions = Tensor::new(array![0.9, 0.2, 0.8].into_dyn(), true);
/// let targets = Tensor::new(array![1.0, 0.0, 1.0].into_dyn(), false);
///
/// let loss = bce_loss(&predictions, &targets);
///
/// // loss = - (1*log(0.9) + 0*log(0.8) + 1*log(0.8))
/// // loss = - ( -0.1053 + 0 + -0.2231 ) = 0.3284
/// assert!((loss.data.borrow().sum() - 0.3284).abs() < 1e-4);
/// ```
pub fn bce_loss(p: &Tensor, y: &Tensor) -> Tensor {
    // Создаем тензор из единиц, чтобы вычислить `1 - y` и `1 - p`.
    // Он не требует градиента.
    let ones = Tensor::ones(p.data.borrow().shape(), false);

    // Первое слагаемое: y * log(p)
    let term1 = y.mul(&p.log());

    // Второе слагаемое: (1 - y) * log(1 - p)
    let term2 = (ones.sub(y)).mul(&(ones.sub(p).log()));

    // Сумма слагаемых: y*log(p) + (1-y)*log(1-p)
    let sum_terms = term1.add(&term2);

    // Усредняем (суммируем) и инвертируем знак.
    // Добавляем небольшой минус перед результатом, чтобы получить -sum(...)
    let neg_one = Tensor::new(ndarray::arr0(-1.0).into_dyn(), false);
    let loss = sum_terms.sum().mul(&neg_one);
    
    loss
}