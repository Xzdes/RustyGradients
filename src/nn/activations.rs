use crate::nn::module::Module;
use crate::tensor::Tensor;

/// Слой активации ReLU (Rectified Linear Unit).
/// Этот слой не имеет обучаемых параметров.
pub struct ReLU;

impl ReLU {
    /// Создает новый экземпляр слоя ReLU.
    pub fn new() -> Self {
        ReLU {}
    }
}

// Реализуем трейт `Default`, чтобы можно было создавать слой через `ReLU::default()`.
// Это необязательно, но является хорошей практикой для структур без полей.
impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ReLU {
    /// Прямой проход просто вызывает математическую операцию ReLU.
    fn forward(&self, inputs: &Tensor) -> Tensor {
        // Мы пока не добавили метод .relu() в Tensor,
        // поэтому вызываем операцию напрямую.
        crate::ops::elementwise::relu_op(inputs)
    }

    /// ReLU не имеет обучаемых параметров, поэтому возвращаем пустой вектор.
    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}