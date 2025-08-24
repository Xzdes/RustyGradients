use crate::nn::module::Module;
use crate::tensor::Tensor;
use ndarray::IxDyn;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

/// Слой встраивания (Embedding Layer).
///
/// Хранит таблицу векторов (встраиваний) и по заданным индексам (ID токенов)
/// возвращает соответствующие векторы.
pub struct Embedding {
    /// Матрица весов, где каждая строка - это вектор-встраивание.
    /// Форма: `[vocab_size, embedding_dim]`.
    pub weights: Tensor,
    pub vocab_size: usize,
    pub embedding_dim: usize,
}

impl Embedding {
    /// Создает новый слой Embedding.
    ///
    /// # Аргументы
    ///
    /// * `vocab_size` - Размер словаря (количество уникальных токенов).
    /// * `embedding_dim` - Размерность вектора-встраивания для каждого токена.
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        // Инициализируем веса небольшими случайными значениями из равномерного распределения.
        let weights_data = ndarray::ArrayD::random(
            IxDyn(&[vocab_size, embedding_dim]),
            Uniform::new(-1.0, 1.0),
        );
        let weights = Tensor::new(weights_data, true);

        Self {
            weights,
            vocab_size,
            embedding_dim,
        }
    }
}

impl Module for Embedding {
    /// Выполняет прямой проход, выбирая векторы по индексам.
    ///
    /// # Аргументы
    ///
    /// * `inputs` - 2D-тензор с целочисленными индексами (ID токенов).
    ///   Форма: `[batch_size, seq_len]`.
    fn forward(&self, inputs: &Tensor) -> Tensor {
        crate::ops::embedding::embedding_op(inputs, &self.weights)
    }

    /// Возвращает матрицу весов как единственный обучаемый параметр.
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone()]
    }
}