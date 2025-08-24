use crate::nn::module::Module;
use crate::tensor::Tensor;

const EPSILON: f32 = 1e-5; // Стандартное значение epsilon для LayerNorm

/// Слой нормализации (Layer Normalization).
///
/// Нормализует активации по оси признаков для каждого элемента в батче.
/// Имеет два обучаемых параметра: `gamma` (масштаб) и `beta` (сдвиг).
pub struct LayerNorm {
    gamma: Tensor,
    beta: Tensor,
    epsilon: f32,
}

impl LayerNorm {
    /// Создает новый слой LayerNorm.
    ///
    /// # Аргументы
    ///
    /// * `normalized_shape` - Размерность признаков, по которой происходит нормализация.
    ///   Например, для Трансформера это `embedding_dim`.
    pub fn new(normalized_shape: usize) -> Self {
        // gamma инициализируется единицами.
        let gamma = Tensor::ones(&[1, normalized_shape], true);
        // beta инициализируется нулями.
        let beta = Tensor::zeros(&[1, normalized_shape], true);

        Self {
            gamma,
            beta,
            epsilon: EPSILON,
        }
    }
}

impl Module for LayerNorm {
    /// Выполняет прямой проход LayerNorm.
    fn forward(&self, inputs: &Tensor) -> Tensor {
        crate::ops::norm::layernorm_op(inputs, &self.gamma, &self.beta, self.epsilon)
    }

    /// Возвращает `gamma` и `beta` как обучаемые параметры.
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}