// tests/nn_test.rs

// Подключаем наш общий модуль
mod common;

// --- ИСПРАВЛЕНИЕ: Используем новое имя крейта ---
use rusty_gradients::nn::{FeedForward, Linear, Module, MultiHeadAttention, ReLU, TransformerBlock};
use rusty_gradients::tensor::Tensor;

use ndarray::ArrayD;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

// Атрибут #[test] помечает эту функцию как тест.
#[test]
fn test_linear_layer_backward() {
    // 1. Arrange (Подготовка)
    let linear_layer = Linear::new(10, 5);
    let input = Tensor::new(
        ArrayD::random(ndarray::IxDyn(&[1, 10]), Uniform::new(-1.0, 1.0)),
        true,
    );

    // 2. Act (Действие)
    let output = linear_layer.forward(&input);
    let loss = output.sum(); // Простейшая функция потерь для теста
    loss.backward();

    // 3. Assert (Проверка)
    common::check_all_grads(&linear_layer, "Linear Layer");
    common::check_input_grad(&input, "Linear Layer Input");
}

#[test]
fn test_relu_layer_backward() {
    // 1. Arrange
    let relu_layer = ReLU::new();
    // Создаем данные, где есть и положительные, и отрицательные значения
    let input_data = ndarray::array![-2.0, -1.0, 0.0, 1.0, 2.0].into_dyn();
    let input = Tensor::new(input_data, true);

    // 2. Act
    let output = relu_layer.forward(&input);
    let loss = output.sum();
    loss.backward();

    // 3. Assert
    // У ReLU нет параметров, так что проверяем только градиент входа
    common::check_input_grad(&input, "ReLU Layer Input");

    // Более точная проверка: градиент должен быть [0, 0, 0, 1, 1]
    let expected_grad = ndarray::array![0.0, 0.0, 0.0, 1.0, 1.0].into_dyn();
    let actual_grad = input.grad.as_ref().unwrap().borrow();
    assert_eq!(*actual_grad, expected_grad, "Градиент для ReLU рассчитан неверно!");
    println!("--> OK: Градиент для ReLU Layer Input корректен.");
}

#[test]
fn test_mha_layer_backward() {
    // 1. Arrange
    let embed_dim = 16;
    let num_heads = 4;
    let batch_size = 2;
    let seq_len = 5;

    let mha = MultiHeadAttention::new(embed_dim, num_heads);
    let input_shape = ndarray::IxDyn(&[batch_size, seq_len, embed_dim]);
    let input = Tensor::new(ArrayD::random(input_shape, Uniform::new(-1.0, 1.0)), true);

    // 2. Act
    let output = mha.forward(&input);
    let loss = output.sum();
    loss.backward();

    // 3. Assert
    common::check_all_grads(&mha, "MultiHeadAttention");
    common::check_input_grad(&input, "MultiHeadAttention Input");
}

#[test]
fn test_feedforward_layer_backward() {
    // 1. Arrange
    let embed_dim = 16;
    let hidden_dim = 32;
    let batch_size = 2;
    let seq_len = 5;

    let ff_layer = FeedForward::new(embed_dim, hidden_dim);
    let input_shape = ndarray::IxDyn(&[batch_size, seq_len, embed_dim]);
    let input = Tensor::new(ArrayD::random(input_shape, Uniform::new(-1.0, 1.0)), true);

    // 2. Act
    let output = ff_layer.forward(&input);
    let loss = output.sum();
    loss.backward();

    // 3. Assert
    common::check_all_grads(&ff_layer, "FeedForward Layer");
    common::check_input_grad(&input, "FeedForward Layer Input");
}

// --- НОВЫЙ ТЕСТ ---
#[test]
fn test_transformer_block_backward() {
    // 1. Arrange
    let embed_dim = 16;
    let num_heads = 4;
    let hidden_dim = 32;
    let batch_size = 2;
    let seq_len = 5;

    let block = TransformerBlock::new(embed_dim, num_heads, hidden_dim);
    let input_shape = ndarray::IxDyn(&[batch_size, seq_len, embed_dim]);
    let input = Tensor::new(ArrayD::random(input_shape, Uniform::new(-1.0, 1.0)), true);

    // 2. Act
    let output = block.forward(&input);
    let loss = output.sum();
    loss.backward();

    // 3. Assert
    common::check_all_grads(&block, "TransformerBlock");
    common::check_input_grad(&input, "TransformerBlock Input");
}