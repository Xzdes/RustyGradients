use crate::nn::{FeedForward, LayerNorm, Module, MultiHeadAttention};
use crate::tensor::Tensor;
use std::ops::Add;

/// Один блок кодировщика Трансформера.
/// Состоит из:
/// 1. Слой Multi-Head Attention с LayerNorm и остаточным соединением.
/// 2. Слой FeedForward с LayerNorm и остаточным соединением.
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    norm1: LayerNorm,
    feed_forward: FeedForward,
    norm2: LayerNorm,
}

impl TransformerBlock {
    /// Создает новый блок Трансформера.
    ///
    /// # Аргументы
    ///
    /// * `embed_dim` - Размерность встраиваний (например, 512).
    /// * `num_heads` - Количество "голов" внимания.
    /// * `ff_hidden_dim` - Размерность скрытого слоя в FeedForward сети.
    pub fn new(embed_dim: usize, num_heads: usize, ff_hidden_dim: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(embed_dim, num_heads),
            norm1: LayerNorm::new(embed_dim),
            feed_forward: FeedForward::new(embed_dim, ff_hidden_dim),
            norm2: LayerNorm::new(embed_dim),
        }
    }
}

impl Module for TransformerBlock {
    /// Прямой проход через блок Трансформера.
    fn forward(&self, inputs: &Tensor) -> Tensor {
        // 1. Под-слой Multi-Head Attention
        // Сначала нормализуем вход, затем пропускаем через attention
        let attention_output = self.attention.forward(&self.norm1.forward(inputs));
        // Первое остаточное соединение (Add)
        let x = inputs.add(&attention_output);

        // 2. Под-слой FeedForward
        // Сначала нормализуем результат первого под-слоя, затем пропускаем через FFN
        let ff_output = self.feed_forward.forward(&self.norm2.forward(&x));
        // Второе остаточное соединение (Add)
        let output = x.add(&ff_output);

        output
    }

    /// Собирает параметры из всех вложенных модулей.
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.attention.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.feed_forward.parameters());
        params.extend(self.norm2.parameters());
        params
    }
}