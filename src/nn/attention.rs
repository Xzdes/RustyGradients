use crate::nn::{Linear, Module};
use crate::tensor::Tensor;
use std::ops::Mul;

/// Реализация многоголового внимания (Multi-Head Self-Attention).
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
}

impl MultiHeadAttention {
    /// Создает новый слой MultiHeadAttention.
    ///
    /// # Аргументы
    /// * `embed_dim` - Размерность встраиваний (входных векторов).
    /// * `num_heads` - Количество "голов" внимания.
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(
            embed_dim % num_heads == 0,
            "embed_dim должен делиться на num_heads без остатка."
        );

        Self {
            num_heads,
            head_dim: embed_dim / num_heads,
            embed_dim,
            w_q: Linear::new(embed_dim, embed_dim),
            w_k: Linear::new(embed_dim, embed_dim),
            w_v: Linear::new(embed_dim, embed_dim),
            w_o: Linear::new(embed_dim, embed_dim),
        }
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        let batch_size = inputs.data.borrow().shape()[0];
        let seq_len = inputs.data.borrow().shape()[1];
        
        // 1. Линейные проекции для Q, K, V
        let q = self.w_q.forward(inputs);
        let k = self.w_k.forward(inputs);
        let v = self.w_v.forward(inputs);

        // 2. Разделение на "головы" и транспонирование
        let q_heads = q
            .reshape(vec![batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k_heads = k
            .reshape(vec![batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let v_heads = v
            .reshape(vec![batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);

        // 3. Вычисление очков внимания
        let scores = q_heads.dot(&k_heads.transpose(2, 3));

        // Масштабирование
        let scale_inv = Tensor::new(
            ndarray::arr0(1.0 / (self.head_dim as f32).sqrt()).into_dyn(),
            false,
        );
        let scores_scaled = scores.mul(&scale_inv);

        // Применение Softmax
        let attention_weights = scores_scaled.softmax();

        // 4. Взвешивание векторов значений (V)
        let attention_output = attention_weights.dot(&v_heads);

        // 5. Слияние голов
        let concatenated_output = attention_output
            .transpose(1, 2)
            .reshape(vec![batch_size, seq_len, self.embed_dim]);
            
        // 6. Выходная линейная проекция
        self.w_o.forward(&concatenated_output)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.w_q.parameters());
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        params
    }
}