// tests/common.rs

// Делаем так, чтобы компилятор не ругался, что эта функция не используется
// внутри `common.rs`. Она будет использоваться в других файлах.
#![allow(dead_code)]

use slmrustai::nn::Module;
use slmrustai::tensor::Tensor;
// --- ИСПРАВЛЕНИЕ: Удаляем неиспользуемый `Ordering` ---

/// Вспомогательная функция для проверки градиентов.
/// Проверяет, что градиенты не нулевые, и выводит их сумму.
pub fn check_all_grads(model: &dyn Module, name: &str) {
    println!("\n--- Проверка градиентов для '{}' ---", name);
    let mut all_zero = true;
    for (i, p) in model.parameters().iter().enumerate() {
        // Убеждаемся, что градиент существует
        let grad_tensor = p.grad.as_ref().expect("Градиент должен существовать после backward()");
        let grad_sum_abs = grad_tensor.borrow().iter().map(|x| x.abs()).sum::<f32>();
        
        if grad_sum_abs > 1e-8 {
            all_zero = false;
        }
        println!("  Параметр #{}: сумма абс. градиента = {}", i, grad_sum_abs);
    }

    if all_zero {
        // Паника в тесте - это провал теста. Это именно то, что нам нужно.
        panic!("ПРОВАЛ: Все градиенты для '{}' нулевые!", name);
    } else {
        println!("--> OK: Градиенты для '{}' успешно вычислены.", name);
    }
}

/// Вспомогательная функция для проверки градиента входного тензора.
pub fn check_input_grad(input_tensor: &Tensor, name: &str) {
    println!("\n--- Проверка градиента для входа '{}' ---", name);
    let grad_tensor = input_tensor.grad.as_ref().expect("Градиент входа должен существовать");
    let grad_sum_abs = grad_tensor.borrow().iter().map(|x| x.abs()).sum::<f32>();
    
    println!("Сумма абс. градиента для входа: {}", grad_sum_abs);
    
    if grad_sum_abs < 1e-8 {
        panic!("ПРОВАЛ: Градиент для входа '{}' нулевой!", name);
    } else {
        println!("--> OK: Градиент для входа '{}' успешно вычислен.", name);
    }
}