// Импортируем нашу новую функцию потерь.
use slmrustai::losses::mse_loss;
use slmrustai::nn::{Linear, Module, ReLU, Sequential, Sigmoid};
use slmrustai::optim::{Adam, Optimizer};
use slmrustai::tensor::Tensor;

fn main() {
    println!("--- Решаем задачу XOR с помощью Adam и модуля losses ---");

    // 1. Создаем модель (без изменений)
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 4)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(4, 1)),
        Box::new(Sigmoid::new()),
    ]);

    // 2. Создаем оптимизатор Adam (без изменений)
    let mut optimizer = Adam::new(model.parameters(), 0.01, None, None);

    // 3. Данные для обучения (XOR) (без изменений)
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
    for epoch in 1..=1000 {
        // Прямой проход
        let y_pred = model.forward(&x);

        // --- ИЗМЕНЕНИЕ: Используем функцию `mse_loss` ---
        let loss = mse_loss(&y_pred, &y_true);

        // Печатаем ошибку
        if epoch % 200 == 0 || epoch == 1 {
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
    
    let final_preds = model.forward(&x);
    println!("\nПредсказанные значения (вероятности):");
    println!("{:?}", final_preds.data.borrow());
    
    println!("\nПредсказанные значения (округленные до 0 или 1):");
    println!("{:?}", final_preds.data.borrow().mapv(|val| val.round()));

    println!("\nИстинные значения:");
    println!("{:?}", y_true.data.borrow());
}