use slmrustai::layers::{Dense, Layer};
use slmrustai::optimizers::{Optimizer, SGD};
use slmrustai::tensor::Tensor;

fn main() {
    println!("--- Начинаем тренировочный цикл ---");

    // 1. Создаем модель (простая сеть с одним слоем)
    // Наша цель - научить модель выполнять операцию XOR.
    // Это нелинейная задача, поэтому один слой справится с ней не идеально,
    // но мы сможем увидеть процесс обучения.
    let model = Dense::new(2, 1); // 2 входа, 1 выход

    // 2. Создаем оптимизатор
    // Мы передаем в него параметры модели и устанавливаем скорость обучения (learning rate).
    let mut optimizer = SGD::new(model.parameters(), 0.1);

    // 3. Данные для обучения (батч из 4-х примеров для XOR)
    let x_data = ndarray::array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ]
    .into_dyn();
    let x = Tensor::new(x_data, false); // Входные данные не требуют градиента

    // Истинные метки для XOR
    let y_true_data = ndarray::array![[0.0], [1.0], [1.0], [0.0]].into_dyn();
    let y_true = Tensor::new(y_true_data, false);

    // 4. Тренировочный цикл
    println!("\n--- Старт обучения ---");
    for epoch in 1..=20 {
        // --- Прямой проход (Forward pass) ---
        // Получаем предсказания модели для наших входных данных `x`.
        let y_pred = model.forward(&x);

        // --- Вычисление ошибки (Loss Calculation) ---
        // Мы используем Среднеквадратичную ошибку (MSE - Mean Squared Error).
        // loss = sum((y_pred - y_true)^2)
        let error = &y_pred - &y_true;
        let loss = error.powf(2.0).sum();

        // Печатаем ошибку каждые несколько эпох, чтобы видеть прогресс.
        if epoch % 5 == 0 || epoch == 1 {
            // .data.borrow() - получаем доступ к данным тензора ошибки.
            println!("Эпоха: {}, Ошибка (Loss): {:?}", epoch, loss.data.borrow());
        }

        // --- Обратный проход (Backward pass) ---

        // Сначала необходимо обнулить градиенты с предыдущего шага.
        // Если этого не сделать, градиенты будут накапливаться.
        optimizer.zero_grad();
        
        // Вычисляем градиенты для всех параметров модели, начиная с `loss`.
        loss.backward();

        // --- Шаг оптимизации (Optimization step) ---
        // Оптимизатор обновляет веса модели, используя вычисленные градиенты.
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