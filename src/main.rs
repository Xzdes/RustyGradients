use slmrustai::nn::{Module, MultiHeadAttention};
use slmrustai::tensor::Tensor;
// --- ИСПРАВЛЕНИЕ: Убираем неиспользуемые импорты ---
// use slmrustai::optim::{Adam, Optimizer};

// Вспомогательная функция для проверки градиентов
fn check_all_grads(model: &dyn Module, name: &str) {
    println!("\n--- Проверка градиентов для '{}' ---", name);
    let mut all_zero = true;
    for (i, p) in model.parameters().iter().enumerate() {
        let grad_sum = p.grad.as_ref().unwrap().borrow().sum().abs();
        if grad_sum > 1e-8 {
            all_zero = false;
        }
        println!("  Параметр #{}: сумма абс. градиента = {}", i, grad_sum);
    }
    if all_zero {
        println!("--> ПРЕДУПРЕЖДЕНИЕ: Все градиенты для '{}' нулевые!", name);
    } else {
        println!("--> OK: Градиенты для '{}' успешно вычислены.", name);
    }
}


fn main() {
    println!("--- Тестируем слой MultiHeadAttention ---");

    // Параметры
    let embed_dim = 16;
    let num_heads = 4;
    let batch_size = 2;
    let seq_len = 5;

    // 1. Создаем слой
    let mha = MultiHeadAttention::new(embed_dim, num_heads);

    // 2. --- ИСПРАВЛЕНИЕ: Правильный способ создания случайного массива ---
    use ndarray::ArrayD;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    let input_shape = ndarray::IxDyn(&[batch_size, seq_len, embed_dim]);
    let input_data = ArrayD::random(input_shape, Uniform::new(-1.0, 1.0));
    
    let input_tensor = Tensor::new(input_data, true);
    
    println!("Форма входного тензора: {:?}", input_tensor.data.borrow().shape());

    // 3. Прямой проход
    let output = mha.forward(&input_tensor);

    println!("Форма выходного тензора: {:?}", output.data.borrow().shape());

    assert_eq!(output.data.borrow().shape(), input_tensor.data.borrow().shape());

    // 4. Обратный проход
    let loss = output.sum();
    
    println!("\n--- Запускаем backward() на выходе ---");
    loss.backward();

    // 5. Проверяем градиенты
    check_all_grads(&mha, "MultiHeadAttention");
    
    println!("\nПроверка входного градиента...");
    let input_grad_sum = input_tensor.grad.as_ref().unwrap().borrow().sum().abs();
    println!("Сумма абс. градиента для входа: {}", input_grad_sum);
    assert!(input_grad_sum > 1e-8, "Градиент для входа не должен быть нулевым!");
    println!("--> OK: Градиент для входа успешно вычислен.");
}