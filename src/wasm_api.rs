// src/wasm_api.rs
use wasm_bindgen::prelude::*;
use crate::models::gpt::{GPTModel, GptConfig};
use crate::optim::{Adam, Optimizer};
use crate::tensor::Tensor;
use crate::losses::cross_entropy_loss;
use serde::{Serialize, Deserialize};
use ndarray::Array;
use rand::Rng;
use crate::nn::Module;

// Утилита для инициализации отладки. Вызовите ее один раз из JS.
#[wasm_bindgen]
pub fn init_panic_hook() {
    #[cfg(feature = "wasm-debug")]
    console_error_panic_hook::set_once();
}

// Структура для удобной сериализации/десериализации весов.
#[derive(Serialize, Deserialize)]
struct SerializableTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[wasm_bindgen]
pub struct WasmGptTrainer {
    model: GPTModel,
    optimizer: Adam,
    config: GptConfig,
}

#[wasm_bindgen]
impl WasmGptTrainer {
    /// Конструктор, вызываемый из JavaScript для создания и инициализации модели.
    #[wasm_bindgen(constructor)]
    pub fn new(
        block_size: usize,
        vocab_size: usize,
        num_layers: usize,
        num_heads: usize,
        embedding_dim: usize,
        learning_rate: f32,
    ) -> Self {
        let config = GptConfig {
            block_size,
            vocab_size,
            num_layers,
            num_heads,
            embedding_dim,
        };
        
        let model = GPTModel::new(config.clone());
        let optimizer = Adam::new(model.parameters(), learning_rate, None, None);
        
        Self { model, optimizer, config }
    }

    /// Выполняет один полный шаг обучения.
    /// Принимает батчи данных в виде плоских массивов из JS.
    /// Возвращает значение функции потерь (loss).
    pub fn train_step(&mut self, x_batch: &[u32], y_batch: &[u32]) -> f32 {
        let batch_size = 1;
        let seq_len = self.config.block_size;

        let x_batch_f32: Vec<f32> = x_batch.iter().map(|&x| x as f32).collect();
        let y_batch_f32: Vec<f32> = y_batch.iter().map(|&y| y as f32).collect();

        let x_tensor = Tensor::new(
            Array::from_shape_vec((batch_size, seq_len), x_batch_f32)
                .expect("Failed to create X tensor from JS data")
                .into_dyn(),
            false,
        );
        let y_tensor = Tensor::new(
            Array::from_shape_vec(y_batch_f32.len(), y_batch_f32)
                .expect("Failed to create Y tensor from JS data")
                .into_dyn(),
            false,
        );
        
        self.optimizer.zero_grad();
        let logits = self.model.forward(&x_tensor).expect("Forward pass failed");
        
        let loss = cross_entropy_loss(&logits, &y_tensor);
        let loss_val = loss.data.borrow().iter().sum();

        loss.backward();
        
        self.clip_gradients(1.0);
        self.optimizer.step();

        loss_val
    }

    /// Генерирует текст, используя сэмплирование Top-K для более качественных результатов.
    pub fn generate(&self, prompt_ids: &[u32], max_new_tokens: usize, temperature: f32, top_k: usize) -> Vec<u32> {
        let mut rng = rand::thread_rng();
        let mut context_ids: Vec<u32> = prompt_ids.to_vec();
        let mut generated_ids: Vec<u32> = Vec::new();

        for _ in 0..max_new_tokens {
            let start = context_ids.len().saturating_sub(self.config.block_size);
            let current_context_slice = &context_ids[start..];
            
            let context_f32: Vec<f32> = current_context_slice.iter().map(|&id| id as f32).collect();
            let context_tensor = Tensor::new(
                Array::from_shape_vec((1, context_f32.len()), context_f32)
                    .unwrap().into_dyn(),
                false,
            );

            let logits_tensor = self.model.forward(&context_tensor).expect("Generation forward pass failed");
            let logits_data = logits_tensor.data.borrow();
            
            let vocab_size = self.config.vocab_size;
            let last_step_logits_start_index = (current_context_slice.len() - 1) * vocab_size;
            let mut last_step_logits: Vec<f32> = logits_data.as_slice().unwrap()[last_step_logits_start_index..].to_vec();

            // --- ФИНАЛЬНОЕ УЛУЧШЕНИЕ: TOP-K SAMPLING ---
            
            // 1. Создаем пары (индекс, значение) и сортируем по значению в убывающем порядке
            let mut indexed_logits: Vec<(usize, f32)> = last_step_logits.iter().cloned().enumerate().collect();
            indexed_logits.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            // 2. Все логиты, которые не вошли в top_k, "удаляем", устанавливая им очень низкое значение
            if top_k > 0 && top_k < indexed_logits.len() {
                for i in top_k..indexed_logits.len() {
                    let original_index = indexed_logits[i].0;
                    last_step_logits[original_index] = f32::NEG_INFINITY;
                }
            }

            // --- КОНЕЦ УЛУЧШЕНИЯ ---

            // Применяем температуру и Softmax к отфильтрованным логитам
            let mut probs: Vec<f32> = last_step_logits.iter()
                .map(|&l| (l / temperature).exp())
                .collect();
            let sum_probs: f32 = probs.iter().sum();
            // Избегаем деления на ноль, если все вероятности стали 0
            if sum_probs > 0.0 {
                probs.iter_mut().for_each(|p| *p /= sum_probs);
            }

            // Сэмплируем следующий токен из нового распределения вероятностей
            let rand_val: f32 = rng.gen();
            let mut cumulative_prob = 0.0;
            let mut next_id = 0;
            for (i, &prob) in probs.iter().enumerate() {
                cumulative_prob += prob;
                if rand_val < cumulative_prob {
                    next_id = i as u32;
                    break;
                }
            }

            context_ids.push(next_id);
            generated_ids.push(next_id);
        }

        generated_ids
    }
    
    /// Сериализует все веса модели в JSON-строку для сохранения.
    #[wasm_bindgen(js_name = getWeightsAsJson)]
    pub fn get_weights_as_json(&self) -> String {
        let params = self.model.parameters();
        let serializable_params: Vec<SerializableTensor> = params.iter().map(|p| {
            let p_data = p.data.borrow();
            SerializableTensor {
                shape: p_data.shape().to_vec(),
                data: p_data.iter().cloned().collect(),
            }
        }).collect();

        serde_json::to_string(&serializable_params).expect("Failed to serialize model weights")
    }

    /// Десериализует веса из JSON и загружает их в модель.
    #[wasm_bindgen(js_name = loadWeightsFromJson)]
    pub fn load_weights_from_json(&mut self, json_str: &str) {
        let loaded_params: Vec<SerializableTensor> = serde_json::from_str(json_str)
            .expect("Failed to deserialize model weights");
        
        let model_params = self.model.parameters();
        
        assert_eq!(loaded_params.len(), model_params.len(), "Mismatch in number of parameter tensors during loading");

        for (param, loaded_param) in model_params.iter().zip(loaded_params) {
            let mut param_data = param.data.borrow_mut();
            assert_eq!(param_data.shape(), loaded_param.shape.as_slice(), "Mismatch in tensor shape during loading");
            
            let new_data = Array::from_shape_vec(loaded_param.shape, loaded_param.data)
                .expect("Failed to create array from loaded data");
            
            param_data.assign(&new_data.into_dyn());
        }
    }

    // Внутренняя вспомогательная функция для обрезки градиентов
    fn clip_gradients(&self, max_norm: f32) {
        let params = self.model.parameters();
        
        let mut total_norm_sq: f32 = 0.0;
        for p in &params {
            if let Some(grad) = &p.grad {
                let grad_data = grad.borrow();
                for &val in grad_data.iter() {
                    total_norm_sq += val * val;
                }
            }
        }
        let total_norm = total_norm_sq.sqrt();

        if total_norm > max_norm {
            let scale_factor = max_norm / total_norm;
            for p in &params {
                if let Some(grad) = &p.grad {
                    grad.borrow_mut().mapv_inplace(|g| g * scale_factor);
                }
            }
        }
    }
}