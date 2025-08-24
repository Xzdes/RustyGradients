use crate::optim::optimizer::Optimizer;
use crate::tensor::Tensor;

/// Оптимизатор стохастического градиентного спуска (SGD).
pub struct SGD {
    parameters: Vec<Tensor>,
    lr: f32, // learning rate
}

impl SGD {
    /// Создает новый экземпляр оптимизатора SGD.
    ///
    /// # Аргументы
    ///
    /// * `parameters` - Список тензоров (параметров модели), которые нужно оптимизировать.
    /// * `lr` - Скорость обучения (learning rate).
    pub fn new(parameters: Vec<Tensor>, lr: f32) -> Self {
        Self { parameters, lr }
    }
}

impl Optimizer for SGD {
    /// Обновляет каждый параметр, используя его градиент.
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
    /// Это необходимо делать перед каждым вызовом `backward()`, чтобы градиенты
    /// не накапливались от итерации к итерации.
    fn zero_grad(&self) {
        for p in &self.parameters {
            if let Some(grad) = &p.grad {
                // Заполняем массив градиентов нулями.
                grad.borrow_mut().fill(0.0);
            }
        }
    }
}