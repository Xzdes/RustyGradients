// Импортируем компоненты, используя новую модульную структуру.
use slmrustai::nn::{Linear, Module};
use slmrustai::optim::{Optimizer, SGD};
use slmrustai::tensor::Tensor;
// --- ИСПРАВЛЕНИЕ: Удаляем неиспользуемый импорт трейтов ---
// use std::ops::{Add, Mul, Sub};

fn main() {
    println!("--- Начинаем тренировочный цикл с новой модульной структурой ---");

    // 1. Создаем модель (переименовали Dense в Linear)
    let model = Linear::new(2, 1);

    // 2. Создаем оптимизатор
    let mut optimizer = SGD::new(model.parameters(), 0.1);

    // 3. Данные для обучения (без изменений)
    let x_data = ndarray::array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ]
    .into_dyn();
    let x = Tensor::new(x_data, false);

    let y_true_data = ndarray::array![[0.0], [1.0], [1.0], [0.0]].into_dyn();
    let y_true = Tensor::new(y_true_data, false);

    // 4. Тренировочный цикл
    println!("\n--- Старт обучения ---");
    for epoch in 1..=20 {
        // Прямой проход
        let y_pred = model.forward(&x);

        // Вычисление ошибки (MSE Loss)
        // Теперь мы используем операторы, реализованные через трейты
        let error = y_pred.sub(&y_true); // Используем метод .sub()
        let loss = error.powf(2.0).sum(); 

        // Печатаем ошибку
        if epoch % 5 == 0 || epoch == 1 {
            println!(
                "Эпоха: {}, Ошибка (Loss): {:?}",
                epoch,
                loss.data.borrow()
            );
        }

        // Обнуляем градиенты
        optimizer.zero_grad();
        
        // Обратное распространение ошибки
        loss.backward();

        // Шаг оптимизатора
        optimizer.step();
    }
    
    println!("\n--- Тренировка завершена ---");
    println!("\nФинальные предсказания модели для входов:");
    println!("{:?}", x.data.borrow());
    println!("\nПредсказанные значения:");
    println!("{:?}", model.forward(&x).data.borrow());
    println!("\nИстинные значения:");
    println!("{:?}", y_true.data.borrow());
}