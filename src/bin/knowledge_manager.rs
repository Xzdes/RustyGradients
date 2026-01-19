// src/bin/knowledge_manager.rs

use chrono::{DateTime, Utc};
use rusty_gradients::models::gpt::{GPTModel, GptConfig};
use rusty_gradients::nn::Module;
use rusty_gradients::optim::{Adam, Optimizer};
use rusty_gradients::tensor::Tensor;
use rusty_gradients::losses::cross_entropy_loss;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;
use rand::Rng;
use rand::distributions::{Distribution, WeightedIndex};

// --- 1. СТРУКТУРЫ ДАННЫХ ДЛЯ ХРАНЕНИЯ В JSON ---

#[derive(Serialize, Deserialize, Clone)]
struct SerializableTensor { shape: Vec<usize>, data: Vec<f32> }
#[derive(Serialize, Deserialize)]
struct Checkpoint { epoch: usize, comment: String, weights: Vec<SerializableTensor> }
#[derive(Serialize, Deserialize)]
struct TrainingRecord { step: usize, loss: f32 }
#[derive(Serialize, Deserialize, Default)] // <-- Добавляем Default
struct ExperimentMetadata { model_type: String, vocab_size: usize, embedding_dim: usize, num_layers: usize, num_heads: usize, learning_rate: f32 }

// ИСПРАВЛЕНИЕ: Добавляем `Default`, чтобы безопасно использовать `std::mem::take`
#[derive(Serialize, Deserialize, Default)]
struct Experiment {
    experiment_id: DateTime<Utc>,
    metadata: ExperimentMetadata,
    training_history: Vec<TrainingRecord>,
    checkpoints: Vec<Checkpoint>,
}

#[derive(Serialize, Deserialize, Default)]
struct KnowledgeBase { experiments: Vec<Experiment> }


// --- 2. ТОКЕНИЗАТОР ---

struct Tokenizer {
    char_to_idx: HashMap<char, usize>,
    idx_to_char: HashMap<usize, char>,
}

impl Tokenizer {
    pub fn new(text: &str) -> Self {
        let chars: HashSet<char> = text.chars().collect();
        let mut sorted_chars: Vec<char> = chars.into_iter().collect();
        sorted_chars.sort();
        let mut char_to_idx = HashMap::new();
        let mut idx_to_char = HashMap::new();
        for (i, &ch) in sorted_chars.iter().enumerate() {
            char_to_idx.insert(ch, i);
            idx_to_char.insert(i, ch);
        }
        Self { char_to_idx, idx_to_char }
    }
    
    pub fn encode(&self, s: &str) -> Vec<usize> { s.chars().map(|c| *self.char_to_idx.get(&c).unwrap_or(&0)).collect() }
    pub fn decode(&self, tokens: &[usize]) -> String { tokens.iter().map(|&i| *self.idx_to_char.get(&i).unwrap_or(&'?')).collect() }
    pub fn vocab_size(&self) -> usize { self.char_to_idx.len() }
}


// --- 3. "МЕНЕДЖЕР ЗНАНИЙ" ---

struct KnowledgeManager {
    model: GPTModel,
    optimizer: Adam,
    config: GptConfig,
    knowledge_base: KnowledgeBase,
    current_experiment: Experiment,
    db_path: String,
    tokenizer: Tokenizer,
    train_data: Vec<usize>,
}

impl KnowledgeManager {
    pub fn new(config: GptConfig, learning_rate: f32, db_path: &str, tokenizer: Tokenizer, train_data: Vec<usize>) -> Self {
        let mut model = GPTModel::new(config.clone());
        let optimizer = Adam::new(model.parameters(), learning_rate, None, None);
        let knowledge_base = if Path::new(db_path).exists() {
            let file_contents = fs::read_to_string(db_path).unwrap_or_default();
            serde_json::from_str(&file_contents).unwrap_or_default()
        } else {
            KnowledgeBase::default()
        };

        if let Some(exp) = knowledge_base.experiments.last().and_then(|e| e.checkpoints.last()) {
            println!("Найден последний чекпоинт. Загружаем веса...");
            Self::load_weights_into_model(&mut model, &exp.weights);
        } else {
            println!("Чекпоинты не найдены. Модель инициализирована случайными весами.");
        }

        let metadata = ExperimentMetadata {
            model_type: "GPT".to_string(),
            vocab_size: config.vocab_size, embedding_dim: config.embedding_dim,
            num_layers: config.num_layers, num_heads: config.num_heads, learning_rate,
        };
        let current_experiment = Experiment {
            experiment_id: Utc::now(),
            metadata, training_history: Vec::new(), checkpoints: Vec::new(),
        };
        Self { model, optimizer, config, knowledge_base, current_experiment, db_path: db_path.to_string(), tokenizer, train_data }
    }

    fn load_weights_into_model(model: &mut GPTModel, loaded_weights: &[SerializableTensor]) {
        let model_params = model.parameters();
        assert_eq!(loaded_weights.len(), model_params.len(), "Ошибка: количество тензоров не совпадает!");
        for (param, loaded_param) in model_params.iter().zip(loaded_weights) {
            let mut param_data = param.data.borrow_mut();
            assert_eq!(param_data.shape(), loaded_param.shape.as_slice(), "Ошибка: форма тензора не совпадает!");
            let new_data = ndarray::Array::from_shape_vec(loaded_param.shape.clone(), loaded_param.data.clone()).unwrap();
            param_data.assign(&new_data.into_dyn());
        }
        println!("--> Веса успешно загружены в модель.");
    }

    pub fn save_knowledge(&mut self) {
        if !self.current_experiment.training_history.is_empty() || !self.current_experiment.checkpoints.is_empty() {
             // ИСПРАВЛЕНИЕ: Заменяем опасный `zeroed()` на безопасный `std::mem::take`
             self.knowledge_base.experiments.push(std::mem::take(&mut self.current_experiment));
        }
        let json_data = serde_json::to_string_pretty(&self.knowledge_base).expect("Ошибка сериализации");
        let mut file = fs::File::create(&self.db_path).expect("Не удалось создать файл");
        file.write_all(json_data.as_bytes()).expect("Ошибка записи");
        println!("База знаний сохранена в '{}'", self.db_path);
    }
    
    fn create_checkpoint(&mut self, step: usize, comment: &str) {
        let params = self.model.parameters();
        let weights = params.iter().map(|p| {
            let p_data = p.data.borrow();
            SerializableTensor { shape: p_data.shape().to_vec(), data: p_data.iter().cloned().collect() }
        }).collect();
        self.current_experiment.checkpoints.push(Checkpoint { epoch: step, comment: comment.to_string(), weights });
        println!("--> Создан чекпоинт на шаге {}", step);
    }

    fn get_batch(&self) -> (Tensor, Tensor) {
        let mut rng = rand::thread_rng();
        let block_size = self.config.block_size;
        let start_index = rng.gen_range(0..=(self.train_data.len() - block_size - 1));
        let x_slice = &self.train_data[start_index..start_index + block_size];
        let y_slice = &self.train_data[start_index + 1..start_index + block_size + 1];
        let x_data: Vec<f32> = x_slice.iter().map(|&t| t as f32).collect();
        let y_data: Vec<f32> = y_slice.iter().map(|&t| t as f32).collect();
        let x_tensor = Tensor::new(ndarray::Array::from_shape_vec((1, block_size), x_data).unwrap().into_dyn(), false);
        let y_tensor = Tensor::new(ndarray::Array::from_shape_vec(block_size, y_data).unwrap().into_dyn(), false);
        (x_tensor, y_tensor)
    }

    pub fn train(&mut self, steps: usize, log_interval: usize, checkpoint_interval: usize) {
        println!("\n--- Начало нового эксперимента: {} ---", self.current_experiment.experiment_id);
        for step in 0..=steps {
            let (x_batch, y_batch) = self.get_batch();
            let logits = self.model.forward(&x_batch).expect("Прямой проход провален");
            let loss = cross_entropy_loss(&logits, &y_batch);
            let loss_val = loss.data.borrow().sum();
            loss.backward();
            self.optimizer.step();
            self.optimizer.zero_grad();
            if step % log_interval == 0 {
                println!("Шаг: {}, Потери: {:.4}", step, loss_val);
                self.current_experiment.training_history.push(TrainingRecord { step, loss: loss_val });
            }
            if step > 0 && step % checkpoint_interval == 0 {
                 self.create_checkpoint(step, "промежуточный");
            }
        }
        // ИСПРАВЛЕНИЕ: Убираем дублирование чекпоинта
        if steps % checkpoint_interval != 0 {
            self.create_checkpoint(steps, "финальный");
        }
        println!("--- Обучение завершено ---");
    }

    pub fn generate(&self, prompt: &str, max_new_tokens: usize, temperature: f32) {
        println!("\n--- Генерация текста ---");
        println!("Затравка: \"{}\"", prompt);
        let mut rng = rand::thread_rng();
        let mut context_tokens = self.tokenizer.encode(prompt);
        for _ in 0..max_new_tokens {
            let context_start = context_tokens.len().saturating_sub(self.config.block_size);
            let context_slice = &context_tokens[context_start..];
            let context_f32: Vec<f32> = context_slice.iter().map(|&t| t as f32).collect();
            let context_tensor = Tensor::new(ndarray::Array::from_shape_vec((1, context_f32.len()), context_f32).unwrap().into_dyn(), false);
            let logits_tensor = self.model.forward(&context_tensor).expect("Ошибка генерации");
            let logits_data = logits_tensor.data.borrow();
            let last_logits_slice = logits_data.slice(ndarray::s![0, -1, ..]);
            let probs: Vec<f32> = last_logits_slice.iter().map(|&l| (l / temperature).exp()).collect();
            let dist = WeightedIndex::new(&probs).unwrap();
            let next_token = dist.sample(&mut rng);
            context_tokens.push(next_token);
            print!("{}", self.tokenizer.decode(&[next_token]));
            std::io::stdout().flush().unwrap();
        }
        println!("\n--- Генерация завершена ---");
    }
}


// --- 4. ТОЧКА ВХОДА В ПРОГРАММУ ---

fn main() {
    let text = fs::read_to_string("input.txt").expect("Ошибка: не удалось прочитать файл 'input.txt'.");
    let tokenizer = Tokenizer::new(&text);
    println!("Данные загружены. Размер словаря: {}, Всего токенов: {}", tokenizer.vocab_size(), tokenizer.encode(&text).len());
    
    let config = GptConfig {
        block_size: 64,
        vocab_size: tokenizer.vocab_size(),
        num_layers: 6,
        num_heads: 6,
        embedding_dim: 192,
    };

    let train_data = tokenizer.encode(&text);
    let mut manager = KnowledgeManager::new(
        config, 1e-3, "knowledge_base.json", tokenizer, train_data
    );

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("\nИспользование:");
        println!("  cargo run --bin knowledge_manager -- train [ШАГИ]");
        println!("  cargo run --bin knowledge_manager -- generate \"ВАША ФРАЗА\" [ДЛИНА]");
        return;
    }

    match args[1].as_str() {
        "train" => {
            let steps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
            manager.train(steps, 100, 500);
            manager.save_knowledge();
        }
        "generate" => {
            let prompt = args.get(2).expect("Нужна фраза-затравка для генерации.");
            let max_tokens = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
            manager.generate(prompt, max_tokens, 0.8);
        }
        _ => { println!("Неизвестная команда: {}", args[1]); }
    }
}