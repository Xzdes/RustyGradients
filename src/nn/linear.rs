//! Модуль, реализующий полносвязный (линейный) слой.

use crate::nn::module::Module;
use crate::tensor::Tensor;
use crate::error::Result;
use ndarray::IxDyn;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

/// Полносвязный (линейный) слой.
///
/// Применяет линейное преобразование к входным данным: `y = xW^T + b`.
/// В нашей реализации, для удобства работы с `ndarray`, формула выглядит как `y = x.dot(W) + b`.
pub struct Linear {
    /// Тензор весов слоя. Форма: `[in_features, out_features]`.
    pub weights: Tensor,
    /// Тензор смещений (bias). Форма: `[1, out_features]`.
    pub bias: Tensor,
}

impl Linear {
    /// Создает новый полносвязный слой.
    ///
    /// # Аргументы
    ///
    /// * `in_features` - Количество входных признаков (размерность входного вектора).
    /// * `out_features` - Количество выходных признаков (количество нейронов в слое).
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // Инициализируем веса небольшими случайными значениями.
        // Используем Kaiming/He инициализацию, которая хорошо подходит для ReLU сетей.
        let limit = (2.0 / in_features as f32).sqrt();
        let weights_data = ndarray::ArrayD::random(
            IxDyn(&[in_features, out_features]),
            Uniform::new(-limit, limit),
        );
        let weights = Tensor::new(weights_data, true);

        // Смещения (bias) инициализируем нулями.
        let bias = Tensor::zeros(&[1, out_features], true);

        Self { weights, bias }
    }
}

impl Module for Linear {
    /// Прямой проход: `output = inputs.dot(weights) + bias`
    ///
    /// `ndarray` поддерживает broadcasting для сложения, поэтому `+ &self.bias`
    /// корректно добавит вектор-строку смещений к каждой строке результата матричного умножения.
    fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        // Пока `dot` и `+` не возвращают `Result`, мы просто выполняем операции
        // и оборачиваем финальный результат в `Ok`.
        let dot_product = inputs.dot(&self.weights);
        let final_output = &dot_product + &self.bias;
        Ok(final_output)
    }

    /// Возвращает веса и смещения как обучаемые параметры слоя.
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}