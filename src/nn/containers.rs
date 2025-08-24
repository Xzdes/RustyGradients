use crate::nn::module::Module;
use crate::tensor::Tensor;

/// Контейнер для последовательного объединения модулей (слоев).
/// Модули добавляются в том порядке, в котором они должны быть выполнены.
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// Создает новый экземпляр `Sequential`.
    ///
    /// # Аргументы
    ///
    /// * `layers` - Вектор "умных указателей" на модули, реализующие трейт `Module`.
    ///   `Box<dyn Module>` позволяет хранить в одном векторе разные типы слоев
    ///   (например, `Linear`, `ReLU` и т.д.).
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }
}

impl Module for Sequential {
    /// Выполняет последовательный прямой проход через все слои.
    /// Выход одного слоя становится входом для следующего.
    fn forward(&self, inputs: &Tensor) -> Tensor {
        // Мы не можем просто переприсваивать `inputs`, так как он заимствован.
        // Поэтому создаем временную переменную `current_output`, которую будем обновлять.
        let mut current_output = inputs.clone();

        for layer in &self.layers {
            current_output = layer.forward(&current_output);
        }

        current_output
    }

    /// Собирает и возвращает параметры из всех вложенных слоев.
    fn parameters(&self) -> Vec<Tensor> {
        // Создаем пустой вектор для сбора параметров.
        let mut params = Vec::new();

        // Проходим по каждому слою и добавляем его параметры в наш общий вектор.
        for layer in &self.layers {
            // `append` перемещает все элементы из вектора параметров слоя в наш `params`.
            params.append(&mut layer.parameters());
        }

        params
    }
}