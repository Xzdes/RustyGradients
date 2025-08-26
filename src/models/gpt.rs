// src/models/gpt.rs
use crate::nn::{Embedding, LayerNorm, Linear, Module, Sequential, TransformerBlock};
use crate::tensor::Tensor;
use crate::error::Result;
use std::ops::{Add};

// Конфигурация модели, которую мы будем передавать из JS.
#[derive(Clone)]
pub struct GptConfig {
    pub block_size: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub embedding_dim: usize,
}

// Структура модели, реализующая трейт Module.
pub struct GPTModel {
    _config: GptConfig,
    token_embedding: Embedding,
    position_embedding: Embedding,
    blocks: Sequential,
    final_ln: LayerNorm,
    output_head: Linear,
}

impl GPTModel {
    pub fn new(config: GptConfig) -> Self {
        let embedding_dim = config.embedding_dim;
        let block_size = config.block_size;
        let vocab_size = config.vocab_size;

        let token_embedding = Embedding::new(vocab_size, embedding_dim);
        let position_embedding = Embedding::new(block_size, embedding_dim);
        
        let transformer_blocks: Vec<Box<dyn Module>> = (0..config.num_layers)
            .map(|_| {
                let block = TransformerBlock::new(embedding_dim, config.num_heads, embedding_dim * 4);
                Box::new(block) as Box<dyn Module>
            })
            .collect();
        let blocks = Sequential::new(transformer_blocks);

        let final_ln = LayerNorm::new(embedding_dim);
        let output_head = Linear::new(embedding_dim, vocab_size);

        Self {
            _config: config,
            token_embedding,
            position_embedding,
            blocks,
            final_ln,
            output_head,
        }
    }
}

impl Module for GPTModel {
    fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let seq_len = idx.data.borrow().shape()[1];
        
        // Получаем эмбеддинги токенов
        let tok_emb = self.token_embedding.forward(idx)?;
        
        // Создаем тензор с позициями [0, 1, 2, ..., seq_len-1]
        let pos_data: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let pos_ids = Tensor::new(
            ndarray::Array::from_shape_vec((1, seq_len), pos_data)
                .expect("Shape creation failed for positional ids")
                .into_dyn(),
            false
        );
        let pos_emb = self.position_embedding.forward(&pos_ids)?;
        
        // Складываем эмбеддинги
        let x = (&tok_emb).add(&pos_emb);
        
        // Прогоняем через блоки трансформера и финальные слои
        let x = self.blocks.forward(&x)?;
        let x = self.final_ln.forward(&x)?;
        let logits = self.output_head.forward(&x)?;
        
        Ok(logits)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.token_embedding.parameters());
        params.extend(self.position_embedding.parameters());
        params.extend(self.blocks.parameters());
        params.extend(self.final_ln.parameters());
        params.extend(self.output_head.parameters());
        params
    }
}