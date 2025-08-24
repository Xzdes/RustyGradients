use slmrustai::nn::{LayerNorm, Module};
use slmrustai::tensor::Tensor;

// Вспомогательная функция для проверки градиентов
fn check_grads(tensor: &Tensor, name: &str) {
    let grad_sum = tensor.grad.as_ref().unwrap().borrow().sum();
    println!("Сумма градиентов для '{}': {}", name, grad_sum);
    if grad_sum.abs() < 1e-6 {
        println!("--> ПРЕДУПРЕЖДЕНИЕ: Градиент для '{}' почти нулевой!", name);
    } else {
        println!("--> OK: Градиент для '{}' успешно вычислен.", name);
    }
}

fn main() {
    println!("--- Тестируем слой LayerNorm ---");

    // Параметры
    let batch_size = 2;
    let features = 4;

    // 1. Создаем слой LayerNorm
    let ln_layer = LayerNorm::new(features);
    let params = ln_layer.parameters();
    let gamma = &params[0];
    let beta = &params[1];

    println!("Gamma (до backward):");
    println!("{:?}", gamma);
    println!("\nBeta (до backward):");
    println!("{:?}", beta);

    // 2. Создаем случайные входные данные
    let input_data = ndarray::Array::from_shape_vec(
        (batch_size, features),
        vec![0.1, 0.5, -0.2, 1.0, -1.5, 0.8, 0.0, 0.3],
    )
    .unwrap()
    .into_dyn();
    let input_tensor = Tensor::new(input_data, true);

    println!("\nВходной тензор:");
    println!("{:?}", input_tensor);

    // 3. Прямой проход
    let output = ln_layer.forward(&input_tensor);
    
    // Чтобы проверить backward, нам нужна скалярная ошибка.
    // Просто просуммируем все выходы.
    let loss = output.sum();

    println!("\nВыход LayerNorm (просуммированный):");
    println!("{:?}", loss.data.borrow());

    // 4. Обратный проход
    println!("\n--- Запускаем backward() на выходе ---");
    loss.backward();

    println!("\n--- Проверяем градиенты после backward() ---\n");
    
    // Проверяем градиенты для обучаемых параметров gamma и beta
    check_grads(gamma, "gamma");
    check_grads(beta, "beta");
    
    // Проверяем градиент для входа
    check_grads(&input_tensor, "input_tensor");
    
    println!("\n--- Детальные градиенты ---");
    println!("Gamma (после backward):");
    println!("{:?}", gamma);
    println!("\nBeta (после backward):");
    println!("{:?}", beta);
    println!("\nВходной тензор (после backward):");
    println!("{:?}", input_tensor);
}