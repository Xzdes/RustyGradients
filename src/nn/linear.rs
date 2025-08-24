use crate::nn::module::Module;
use crate::tensor::Tensor;
use ndarray::IxDyn;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

/// Полносвязный (линейный) слой.
/// Применяет линейное преобразование к входным данным: `y = xW^T + b`.
/// В нашем случае, для удобства, `y = x.dot(W) + b`.
pub struct Linear {
    pub weights: Tensor,
    pub bias: Tensor,
}

impl Linear {
    /// Создает новый полносвязный слой.
    ///
    /// # Аргументы
    ///
    /// * `in_features` - Количество входных признаков.
    /// * `out_features` - Количество выходных признаков (нейронов в слое).
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
    fn forward(&self, inputs: &Tensor) -> Tensor {
        // ndarray поддерживает broadcasting для сложения, поэтому `+ &self.bias`
        // корректно добавит вектор-строку смещений к каждой строке результата.
        &inputs.dot(&self.weights) + &self.bias
    }

    /// Возвращает веса и смещения как обучаемые параметры слоя.
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}