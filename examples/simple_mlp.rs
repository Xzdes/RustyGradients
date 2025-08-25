// --- ИЗМЕНЕНИЕ: Импортируем Adam ---
use rusty_gradients::nn::{Linear, Module, ReLU, Sequential};
use rusty_gradients::optim::{Adam, Optimizer};
use rusty_gradients::tensor::Tensor;
use rusty_gradients::losses::mse_loss;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Пример обучения простой MLP на задаче XOR ---");

    // 1. Создаем данные для задачи XOR
    // XOR: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
    let training_data = Tensor::new(
        ndarray::array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].into_dyn(),
        false,
    );
    let training_labels = Tensor::new(
        ndarray::array![[0.0], [1.0], [1.0], [0.0]].into_dyn(),
        false,
    );

    // 2. Создаем модель: двухслойный перцептрон
    // 2 входа -> 4 скрытых нейрона -> 1 выход
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 4)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(4, 1)),
    ]);

    // 3. Создаем оптимизатор
    // --- ИЗМЕНЕНИЕ: Заменяем SGD на Adam ---
    let mut optimizer = Adam::new(model.parameters(), 0.01, None, None); // lr = 0.01

    // 4. Цикл обучения
    let epochs = 1000;
    for epoch in 0..=epochs {
        // Прямой проход
        let predictions = model.forward(&training_data)?;

        // Вычисление функции потерь
        let loss = mse_loss(&predictions, &training_labels);

        // Обратное распространение ошибки
        loss.backward();

        // Шаг оптимизатора (обновление весов)
        optimizer.step();

        // Обнуление градиентов
        optimizer.zero_grad();

        if epoch % 100 == 0 {
            println!("Эпоха: {}, Потери: {:.4}", epoch, loss.data.borrow().sum());
        }
    }

    // 5. Тестирование модели после обучения
    println!("\n--- Результаты после обучения ---");
    let final_predictions = model.forward(&training_data)?;
    for (i, input_row) in training_data.data.borrow().outer_iter().enumerate() {
        println!(
            "Вход: {}, Предсказание: {:.4}",
            input_row,
            final_predictions.data.borrow()[[i, 0]]
        );
    }

    Ok(())
}