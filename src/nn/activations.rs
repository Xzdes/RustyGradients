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

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ReLU {
    /// Прямой проход просто вызывает математическую операцию ReLU.
    fn forward(&self, inputs: &Tensor) -> Tensor {
        // Мы уже добавили метод .relu() в Tensor.
        inputs.relu()
    }

    /// ReLU не имеет обучаемых параметров, поэтому возвращаем пустой вектор.
    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

// --- НОВЫЙ СЛОЙ: Sigmoid ---

/// Слой активации Sigmoid.
/// "Сжимает" входные значения в диапазон (0, 1), что полезно для
/// интерпретации выходов как вероятностей.
pub struct Sigmoid;

impl Sigmoid {
    /// Создает новый экземпляр слоя Sigmoid.
    pub fn new() -> Self {
        Sigmoid {}
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sigmoid {
    /// Прямой проход вызывает математическую операцию Sigmoid.
    fn forward(&self, inputs: &Tensor) -> Tensor {
        // Мы добавим метод .sigmoid() в Tensor на следующем шаге.
        crate::ops::elementwise::sigmoid_op(inputs)
    }

    /// Sigmoid не имеет обучаемых параметров.
    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}