Абсолютно! Проект достиг той точки, где ему нужна витрина, демонстрирующая всю его мощь и гибкость. `README` — это идеальное место для этого.

Вот капитально переработанный `README.md`, который позиционирует проект как многоцелевой инструмент: мощный CLI для исследователей, гибкая библиотека для Rust-разработчиков и высокопроизводительный WASM-модуль для веб-приложений.

Просто скопируйте весь этот текст и замените им содержимое вашего файла `README.md`.

---

# Rusty Gradients: A Full-Stack AI Toolkit in Rust

**Rusty Gradients** — это комплексный, высокопроизводительный фреймворк для глубокого обучения, написанный на Rust. Он предоставляет все инструменты для создания, обучения и развертывания моделей-трансформеров (GPT), объединяя в себе три ключевых продукта:

1.  **Интерактивный CLI**: Полноценное приложение для проведения AI-экспериментов прямо из командной строки.
2.  **Модуль WebAssembly (WASM)**: Высокопроизводительное ядро для запуска моделей машинного обучения непосредственно в браузере.
3.  **Гибкая библиотека (крейт)**: Мощная основа для создания кастомных нейронных сетей в ваших собственных Rust-проектах.

Проект построен вокруг безопасного и эффективного движка автоматического дифференцирования (autograd), что делает его надежным и быстрым решением для любых задач.

## Ключевые особенности

*   🚀 **Многоцелевое развертывание**: Используйте один и тот же код для CLI-инструментов, веб-приложений (через WASM) и бэкенда на Rust.
*   🧠 **End-to-End GPT**: Обучайте полноценную GPT-модель на ваших текстовых данных и генерируйте текст.
*   📦 **Персистентная База Знаний**: Автоматическое сохранение экспериментов, истории обучения и весов модели в файл `knowledge_base.json`.
*   ⏳ **Возобновляемое обучение**: Прерывайте и продолжайте длительные сеансы обучения без потери прогресса. Модель автоматически загрузит последний чекпоинт.
*   ⚛️ **Ядро Autograd**: В основе лежит надежный движок автоматического дифференцирования с динамическим графом вычислений.
*   🧩 **Современная архитектура**: Включает все необходимые строительные блоки для Трансформеров: `Linear`, `LayerNorm`, `MultiHeadAttention`, `Embedding`.

---

## Вариант 1: Использование как CLI-инструмента для AI-экспериментов

Это основной и самый простой способ использовать проект. Вы сможете обучать модель на тексте и генерировать новый без написания единой строки кода.

### Настройка

1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/Xzdes/RustyGradients
    cd RustyGradients
    ```

2.  **Подготовьте данные для обучения:**
    Создайте файл `input.txt` в корне проекта и поместите в него текст, на котором будет учиться модель. Для старта подойдет небольшой текст на английском.

### Использование

#### Обучение модели
Используйте команду `train`, чтобы начать обучение. Прогресс будет автоматически сохраняться.

```bash
# Запустить обучение на 10 000 шагов
cargo run --bin knowledge_manager -- train 10000
```
Если вы прервете процесс (`Ctrl+C`), просто запустите команду снова, и обучение продолжится с последнего сохраненного чекпоинта.

#### Генерация текста
После обучения используйте команду `generate`, чтобы увидеть, чему научилась модель.

```bash
# Сгенерировать 200 символов, начиная с фразы "There should be"
cargo run --bin knowledge_manager -- generate "There should be" 200
```

---

## Вариант 2: Сборка и использование в WebAssembly (WASM)

`Rusty Gradients` может быть скомпилирован в высокопроизводительный WASM-модуль для запуска нейронных сетей прямо в браузере.

### Настройка

1.  **Установите `wasm-pack`:**
    ```bash
    cargo install wasm-pack
    ```

2.  **Соберите WASM-пакет:**
    В корневой папке проекта выполните команду:
    ```bash
    wasm-pack build --target web
    ```
    Эта команда создаст папку `pkg`, содержащую скомпилированный `.wasm` файл и JavaScript-обертку для него.

### Использование в JavaScript

Скопируйте папку `pkg` в ваш веб-проект и используйте ее следующим образом:

```javascript
// main.js
import init, { WasmGptTrainer, init_panic_hook } from './pkg/rusty_gradients.js';

async function run() {
    // 1. Инициализируем WASM-модуль
    await init();
    
    // Включаем вывод ошибок Rust в консоль браузера (опционально)
    init_panic_hook();

    // 2. Создаем экземпляр тренера с нужной конфигурацией
    console.log("Creating GPT model in WASM...");
    const config = {
        blockSize: 32,
        vocabSize: 65, // Укажите размер вашего словаря
        numLayers: 4,
        numHeads: 4,
        embeddingDim: 64,
        learningRate: 0.001
    };
    const trainer = new WasmGptTrainer(
        config.blockSize,
        config.vocabSize,
        config.numLayers,
        config.numHeads,
        config.embeddingDim,
        config.learningRate
    );

    // 3. Обучаем модель (данные должны быть Uint32Array)
    const xBatch = new Uint32Array([10, 20, 30]); // Пример батча
    const yBatch = new Uint32Array([20, 30, 31]);
    const loss = trainer.train_step(xBatch, yBatch);
    console.log(`Training step finished. Loss: ${loss}`);

    // 4. Генерируем текст
    const prompt = new Uint32Array([1, 2, 3]);
    const generatedTokens = trainer.generate(prompt, 100, 0.8, 10); // prompt, max_tokens, temperature, top_k
    console.log("Generated tokens:", generatedTokens);

    // 5. Сохраняем и загружаем веса
    const weightsJson = trainer.getWeightsAsJson();
    // ... можно сохранить weightsJson в localStorage ...
    // trainer.loadWeightsFromJson(weightsJson);
}

run();
```

---

## Вариант 3: Использование как библиотеки в Rust

Вы можете использовать `rusty-gradients` как зависимость для построения собственных нейронных сетей.

### Настройка
Добавьте крейт в ваш `Cargo.toml`:
```toml
[dependencies]
rusty-gradients = { git = "https://github.com/Xzdes/RustyGradients" }
# или
rusty-gradients = "0.1.0"
```

### Пример использования

Вот как можно обучить простую MLP-сеть на задаче XOR:
```rust
use rusty_gradients::nn::{Linear, Module, ReLU, Sequential};
use rusty_gradients::optim::{Adam, Optimizer};
use rusty_gradients::tensor::Tensor;
use rusty_gradients::losses::mse_loss;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Данные для задачи XOR
    let training_data = Tensor::new(
        ndarray::array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].into_dyn(),
        false,
    );
    let training_labels = Tensor::new(
        ndarray::array![[0.0], [1.0], [1.0], [0.0]].into_dyn(),
        false,
    );

    // 2. Создаем модель
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 4)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(4, 1)),
    ]);

    // 3. Создаем оптимизатор
    let mut optimizer = Adam::new(model.parameters(), 0.01, None, None);

    // 4. Цикл обучения
    for epoch in 0..=1000 {
        let predictions = model.forward(&training_data)?;
        let loss = mse_loss(&predictions, &training_labels);
        loss.backward();
        optimizer.step();
        optimizer.zero_grad();
        if epoch % 100 == 0 {
            println!("Эпоха: {}, Потери: {:.4}", epoch, loss.data.borrow().sum());
        }
    }
    Ok(())
}
```

## План развития

1.  **Оптимизация производительности**: Замена ручных циклов на параллельные вычисления с `rayon`.
2.  **Улучшение токенизатора**: Переход от символьного к Byte Pair Encoding (BPE).
3.  **Гибкая конфигурация CLI**: Вынос гиперпараметров модели в конфигурационный файл (`config.toml`).

## Лицензия

Этот проект распространяется под [лицензией MIT](LICENSE).