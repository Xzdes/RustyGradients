// src/nn/feedforward.rs

use crate::nn::{Linear, Module, ReLU};
use crate::tensor::Tensor;

/// Слой FeedForward, стандартный компонент блока Трансформера.
/// Состоит из двух линейных слоев с активацией ReLU между ними.
/// FFN(x) = ReLU(x * W1 + b1) * W2 + b2
pub struct FeedForward {
    linear1: Linear,
    relu: ReLU,
    linear2: Linear,
}

impl FeedForward {
    /// Создает новый слой FeedForward.
    ///
    /// # Аргументы
    ///
    /// * `embed_dim` - Размерность входных и выходных векторов (например, 512).
    /// * `hidden_dim` - Размерность скрытого слоя (обычно в 4 раза больше `embed_dim`, например, 2048).
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Self {
        Self {
            linear1: Linear::new(embed_dim, hidden_dim),
            relu: ReLU::new(),
            linear2: Linear::new(hidden_dim, embed_dim),
        }
    }
}

impl Module for FeedForward {
    /// Прямой проход: Linear -> ReLU -> Linear
    fn forward(&self, inputs: &Tensor) -> Tensor {
        let x = self.linear1.forward(inputs);
        let x = self.relu.forward(&x);
        self.linear2.forward(&x)
    }

    /// Собирает параметры из обоих вложенных линейных слоев.
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params
    }
}