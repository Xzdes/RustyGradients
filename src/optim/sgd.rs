//! Модуль, реализующий оптимизатор стохастического градиентного спуска (SGD).

use crate::optim::optimizer::Optimizer;
use crate::tensor::Tensor;

/// Оптимизатор стохастического градиентного спуска (SGD).
///
/// Это один из самых простых и фундаментальных алгоритмов оптимизации.
/// Он обновляет параметры модели, двигаясь в направлении, противоположном
/// градиенту функции потерь.
///
/// # Примеры
///
/// ```
/// # use rusty_gradients::nn::{Linear, Module};
/// # use rusty_gradients::optim::{Optimizer, SGD};
/// # use rusty_gradients::tensor::Tensor;
/// // Создаем простую модель
/// let model = Linear::new(10, 1);
/// // Получаем ее параметры
/// let params = model.parameters();
/// // Создаем оптимизатор SGD
/// let mut optim = SGD::new(params, 0.01);
/// ```
pub struct SGD {
    /// Список тензоров (параметров модели), которые нужно оптимизировать.
    parameters: Vec<Tensor>,
    /// Скорость обучения (learning rate).
    lr: f32,
}

impl SGD {
    /// Создает новый экземпляр оптимизатора SGD.
    ///
    /// # Аргументы
    ///
    /// * `parameters` - Вектор тензоров (параметров модели), которые будут обновляться.
    /// * `lr` - Скорость обучения (learning rate). Определяет размер шага при обновлении весов.
    pub fn new(parameters: Vec<Tensor>, lr: f32) -> Self {
        Self { parameters, lr }
    }
}

impl Optimizer for SGD {
    /// Обновляет каждый параметр, используя его градиент.
    ///
    /// Формула: `parameter.data = parameter.data - learning_rate * parameter.grad`
    fn step(&mut self) {
        for p in &self.parameters {
            // Убеждаемся, что у параметра есть градиент.
            if let Some(grad) = &p.grad {
                // Заимствуем данные градиента для чтения.
                let grad_borrow = grad.borrow();
                // Вычисляем величину обновления.
                let update = &*grad_borrow * self.lr;
                // Заимствуем данные параметра для записи и вычитаем обновление.
                *p.data.borrow_mut() -= &update;
            }
        }
    }

    /// Обнуляет градиенты для всех параметров, которые отслеживает оптимизатор.
    fn zero_grad(&self) {
        for p in &self.parameters {
            if let Some(grad) = &p.grad {
                // Заполняем массив градиентов нулями.
                grad.borrow_mut().fill(0.0);
            }
        }
    }
}