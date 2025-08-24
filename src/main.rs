// Импортируем все необходимые компоненты, включая Sigmoid.
use slmrustai::nn::{Linear, Module, ReLU, Sequential, Sigmoid};
use slmrustai::optim::{Optimizer, SGD};
use slmrustai::tensor::Tensor;

fn main() {
    println!("--- Решаем задачу XOR с помощью многослойной нейросети ---");

    // 1. Создаем модель с помощью `Sequential`.
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 4)), // Входной слой: 2 входа -> 4 скрытых нейрона
        Box::new(ReLU::new()),       // Нелинейная функция активации
        Box::new(Linear::new(4, 1)), // Выходной слой: 4 скрытых -> 1 выход
        // --- НОВЫЙ СЛОЙ: Добавляем Sigmoid, чтобы выход был в диапазоне (0, 1) ---
        Box::new(Sigmoid::new()),
    ]);

    // 2. Создаем оптимизатор.
    // Увеличим learning rate, чтобы помочь модели быстрее найти решение.
    let mut optimizer = SGD::new(model.parameters(), 0.5);

    // 3. Данные для обучения (XOR)
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

    // 4. Тренировочный цикл. Увеличим количество эпох, чтобы дать модели время сойтись.
    println!("\n--- Старт обучения ---");
    for epoch in 1..=3000 {
        // Прямой проход
        let y_pred = model.forward(&x);

        // Вычисление ошибки (MSE Loss)
        let error = y_pred.sub(&y_true);
        let loss = error.powf(2.0).sum();

        // Печатаем ошибку реже.
        if epoch % 500 == 0 || epoch == 1 {
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