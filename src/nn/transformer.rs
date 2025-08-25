//! Модуль, реализующий один блок кодировщика Трансформера.

use crate::nn::{FeedForward, LayerNorm, Module, MultiHeadAttention};
use crate::tensor::Tensor;
// --- ИЗМЕНЕНИЕ: Импортируем наш Result ---
use crate::error::Result;
use std::ops::Add;

/// Один блок кодировщика Трансформера.
///
/// Этот блок является основной строительной единицей для большинства
/// современных NLP-моделей. Он состоит из двух основных под-слоев:
/// 1. Слой Multi-Head Attention, который выполняет self-attention.
/// 2. Слой FeedForward, который является простой полносвязной сетью.
///
/// Вокруг каждого из этих под-слоев применяется LayerNorm и остаточное соединение (`Add`).
/// Такая архитектура (pre-normalization) стабилизирует обучение глубоких Трансформеров.
pub struct TransformerBlock {
    /// Слой многоголового внимания.
    attention: MultiHeadAttention,
    /// Слой нормализации перед attention.
    norm1: LayerNorm,
    /// Полносвязная сеть.
    feed_forward: FeedForward,
    /// Слой нормализации перед feed-forward.
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
    ///
    /// Логика: `x + Attention(Norm(x))` -> `x + FeedForward(Norm(x))`
    // --- ИЗМЕНЕНИЕ: Сигнатура функции обновлена ---
    fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        // --- ВРЕМЕННАЯ МЕРА: Используем `.unwrap()` ---
        // Позже, когда все `forward` будут возвращать `Result`,
        // мы заменим `.unwrap()` на `?`.

        // 1. Под-слой Multi-Head Attention
        // Сначала нормализуем вход, затем пропускаем через attention
        let normed_inputs1 = self.norm1.forward(inputs).unwrap();
        let attention_output = self.attention.forward(&normed_inputs1).unwrap();
        // Первое остаточное соединение (Add)
        let x = inputs.add(&attention_output);

        // 2. Под-слой FeedForward
        // Сначала нормализуем результат первого под-слоя, затем пропускаем через FFN
        let normed_inputs2 = self.norm2.forward(&x).unwrap();
        let ff_output = self.feed_forward.forward(&normed_inputs2).unwrap();
        // Второе остаточное соединение (Add)
        let final_output = x.add(&ff_output);

        // --- ИЗМЕНЕНИЕ: Финальный результат оборачивается в Ok() ---
        Ok(final_output)
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