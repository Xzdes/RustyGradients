use slmrustai::nn::{Embedding, Module};
use slmrustai::tensor::Tensor;

fn main() {
    println!("--- Тестируем слой Embedding ---");

    // Параметры
    let vocab_size = 10;    // Размер словаря (10 уникальных слов)
    let embedding_dim = 4;  // Размерность вектора для каждого слова
    let batch_size = 2;     // Два предложения в батче
    let seq_len = 3;        // По три слова в каждом предложении

    // 1. Создаем слой Embedding
    let embedding_layer = Embedding::new(vocab_size, embedding_dim);
    println!("Матрица весов Embedding (до backward):");
    // .parameters() возвращает Vec<Tensor>, берем первый (и единственный) элемент
    println!("{:?}", embedding_layer.parameters()[0]);

    // 2. Создаем входные данные - батч с ID токенов.
    // Это два "предложения": [1, 5, 0] и [8, 3, 5]
    let input_ids_data = ndarray::array![
        [1.0, 5.0, 0.0],
        [8.0, 3.0, 5.0]
    ].into_dyn();
    // Входные ID не требуют градиента
    let input_ids = Tensor::new(input_ids_data, false);
    
    println!("\nВходные ID токенов (форма [{}, {}]):", batch_size, seq_len);
    println!("{:?}", input_ids.data.borrow());

    // 3. Прямой проход
    let output = embedding_layer.forward(&input_ids);

    println!("\nВыход Embedding слоя (форма [{}, {}, {}]):", batch_size, seq_len, embedding_dim);
    println!("{:?}", output.data.borrow());

    // 4. Обратный проход
    // Чтобы проверить градиенты, мы вызовем backward() на выходе.
    // Это сымитирует приход градиента от последующих слоев сети.
    println!("\n--- Запускаем backward() на выходе ---");
    output.backward();

    println!("\nМатрица весов Embedding (после backward):");
    println!("{:?}", embedding_layer.parameters()[0]);
    
    println!("\nПроверка: градиент должен появиться только у тех строк,");
    println!("которые соответствуют ID во входных данных (0, 1, 3, 5, 8).");
    println!("Причем у строки 5 градиент должен быть вдвое больше, так как ID 5 встречался дважды.");
}