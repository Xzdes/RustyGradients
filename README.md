# Rusty Gradients

Rusty Gradients — это фундаментальный фреймворк для глубокого обучения на Rust, построенный вокруг динамического графа вычислений и автоматического дифференцирования (autograd).

Проект создан с целью предоставить ясный, надежный и расширяемый набор инструментов для создания и обучения нейронных сетей, уделяя особое внимание безопасности и контролю, которые предоставляет язык Rust.

## Ключевые особенности

*   **Динамический граф вычислений**: Легко создавайте сложные, изменяемые архитектуры моделей. `Tensor` автоматически отслеживает все операции для построения графа.
*   **Автоматическое дифференцирование**: Вызовите `.backward()` на скалярном тензоре (например, на результате функции потерь), чтобы автоматически рассчитать градиенты для всех обучаемых параметров.
*   **Богатая библиотека слоев**: Включает в себя все необходимые компоненты для современных архитектур: `Linear`, `ReLU`, `LayerNorm`, `Embedding`, `MultiHeadAttention`, `TransformerBlock` и другие.
*   **Современные оптимизаторы**: Встроенные реализации `SGD` и `Adam` для эффективного обучения моделей.
*   **Надежная обработка ошибок**: API, основанный на типе `Result`, позволяет элегантно обрабатывать потенциальные ошибки (например, несовпадение размерностей тензоров) без паники.

## Установка

Добавьте библиотеку в ваш `Cargo.toml`:

```toml
[dependencies]
rusty-gradients = "0.1.0" # Укажите актуальную версию с crates.io
```

## Быстрый старт: Обучение MLP на задаче XOR

Этот пример демонстрирует полный цикл обучения простой нейронной сети для решения классической нелинейной задачи XOR.

```rust
use rusty_gradients::nn::{Linear, Module, ReLU, Sequential};
use rusty_gradients::optim::{Adam, Optimizer};
use rusty_gradients::tensor::Tensor;
use rusty_gradients::losses::mse_loss;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Определяем данные для задачи XOR
    let training_data = Tensor::new(
        ndarray::array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].into_dyn(),
        false,
    );
    let training_labels = Tensor::new(
        ndarray::array![[0.0], [1.0], [1.0], [0.0]].into_dyn(),
        false,
    );

    // 2. Создаем модель: 2 входа -> 4 скрытых нейрона -> 1 выход
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 4)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(4, 1)),
    ]);

    // 3. Создаем оптимизатор Adam
    let mut optimizer = Adam::new(model.parameters(), 0.01, None, None);

    // 4. Запускаем цикл обучения
    for epoch in 0..=1000 {
        // Прямой проход
        let predictions = model.forward(&training_data)?;
        // Вычисление функции потерь
        let loss = mse_loss(&predictions, &training_labels);

        // Обратное распространение ошибки
        loss.backward();
        // Обновление весов
        optimizer.step();
        // Обнуление градиентов
        optimizer.zero_grad();

        if epoch % 100 == 0 {
            println!("Эпоха: {}, Потери: {:.4}", epoch, loss.data.borrow().sum());
        }
    }

    // 5. Проверяем результат после обучения
    let final_predictions = model.forward(&training_data)?;
    println!("\nРезультаты после обучения:");
    println!("{}", final_predictions.data.borrow());

    Ok(())
}
```

## Лицензия

Этот проект распространяется под [лицензией MIT](LICENSE).