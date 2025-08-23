use crate::tensor::Tensor;
// --- ДОБАВЛЯЕМ IxDyn В ИМПОРТЫ ---
use ndarray::{ArrayD, IxDyn};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

/// Трейт, определяющий общий интерфейс для всех слоев.
pub trait Layer {
    /// Выполняет прямой проход через слой.
    fn forward(&self, inputs: &Tensor) -> Tensor;

    /// Возвращает список обучаемых параметров слоя.
    fn parameters(&self) -> Vec<Tensor>;
}

/// Полносвязный слой (Dense Layer).
pub struct Dense {
    pub weights: Tensor,
    pub bias: Tensor,
}

impl Dense {
    /// Создает новый полносвязный слой.
    ///
    /// # Аргументы
    ///
    /// * `in_features` - Количество входных признаков.
    /// * `out_features` - Количество выходных признаков (нейронов в слое).
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // Инициализируем веса небольшими случайными значениями.
        let limit = (2.0 / in_features as f32).sqrt();
        
        // --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
        // Оборачиваем срез с измерениями в конструктор IxDyn()
        let weights_data = ArrayD::random(
            IxDyn(&[in_features, out_features]),
            Uniform::new(-limit, limit),
        );
        let weights = Tensor::new(weights_data, true);

        // Смещения (bias) инициализируем нулями.
        // Здесь все было правильно, т.к. `zeros` уже работает со срезами.
        let bias = Tensor::zeros(&[1, out_features], true);

        Self { weights, bias }
    }
}

impl Layer for Dense {
    /// Прямой проход: `output = inputs.dot(weights) + bias`
    fn forward(&self, inputs: &Tensor) -> Tensor {
        let matmul_result = inputs.dot(&self.weights);
        // ndarray поддерживает broadcasting для сложения, поэтому `+ &self.bias`
        // корректно добавит вектор-строку смещений к каждой строке результата.
        &matmul_result + &self.bias
    }

    /// Возвращает веса и смещения как параметры слоя.
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}