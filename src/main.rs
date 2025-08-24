// Импортируем все необходимые компоненты.
use slmrustai::nn::{Linear, Module, ReLU, Sequential};
use slmrustai::optim::{Optimizer, SGD};
use slmrustai::tensor::Tensor;

fn main() {
    println!("--- Решаем задачу XOR с помощью многослойной нейросети ---");

    // 1. Создаем модель с помощью `Sequential`.
    //    `Box::new()` упаковывает наши слои в "умные указатели".
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 4)), // Входной слой: 2 входа -> 4 скрытых нейрона
        Box::new(ReLU::new()),       // Нелинейная функция активации
        Box::new(Linear::new(4, 1)), // Выходной слой: 4 скрытых -> 1 выход
    ]);

    // 2. Создаем оптимизатор.
    //    Обратите внимание, `model.parameters()` теперь рекурсивно собирает
    //    параметры из обоих `Linear` слоев.
    //    Для более сложной задачи возьмем learning rate поменьше.
    let mut optimizer = SGD::new(model.parameters(), 0.1);

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

    // 4. Тренировочный цикл. Увеличим количество эпох.
    println!("\n--- Старт обучения ---");
    for epoch in 1..=2000 {
        // Прямой проход
        let y_pred = model.forward(&x);

        // Вычисление ошибки (MSE Loss)
        let error = y_pred.sub(&y_true);
        let loss = error.powf(2.0).sum();

        // Печатаем ошибку реже, чтобы не засорять консоль.
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
    println!("\nПредсказанные значения (сырые выходы):");
    println!("{:?}", final_preds.data.borrow());
    
    println!("\nПредсказанные значения (округленные до 0 или 1):");
    println!("{:?}", final_preds.data.borrow().mapv(|val| val.round()));

    println!("\nИстинные значения:");
    println!("{:?}", y_true.data.borrow());
}