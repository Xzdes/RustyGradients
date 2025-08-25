//! Модуль, реализующий оптимизатор Adam.

use crate::optim::optimizer::Optimizer;
use crate::tensor::Tensor;

/// Оптимизатор Adam (Adaptive Moment Estimation).
///
/// Adam - это адаптивный алгоритм оптимизации, который вычисляет индивидуальные
/// скорости обучения для разных параметров. Он хранит экспоненциально затухающие
/// скользящие средние прошлых градиентов (`m`) и их квадратов (`v`).
///
/// Это один из самых популярных и эффективных оптимизаторов для обучения
/// глубоких нейронных сетей.
///
/// # Примеры
///
/// ```
/// # use rusty_gradients::nn::{Linear, Module};
/// # use rusty_gradients::optim::{Adam, Optimizer};
/// # use rusty_gradients::tensor::Tensor;
/// // Создаем модель
/// let model = Linear::new(10, 1);
/// // Создаем оптимизатор Adam с параметрами по умолчанию
/// let mut optim = Adam::new(model.parameters(), 0.001, None, None);
/// ```
pub struct Adam {
    parameters: Vec<Tensor>,
    lr: f32,      // Скорость обучения (learning rate)
    beta1: f32,   // Коэффициент затухания для первого момента
    beta2: f32,   // Коэффициент затухания для второго момента
    epsilon: f32, // Малая константа для численной стабильности
    t: i32,       // Счетчик шагов (времени)

    // Буферы для хранения скользящих средних
    m: Vec<Tensor>, // Первый момент (оценка среднего градиента)
    v: Vec<Tensor>, // Второй момент (оценка нецентрированной дисперсии градиента)
}

impl Adam {
    /// Создает новый экземпляр оптимизатора Adam.
    ///
    /// # Аргументы
    ///
    /// * `parameters` - Параметры модели для оптимизации.
    /// * `lr` - Скорость обучения (рекомендуемое значение: 0.001).
    /// * `betas` - Кортеж `(beta1, beta2)`. По умолчанию `(0.9, 0.999)`.
    /// * `epsilon` - Малая добавка для предотвращения деления на ноль. По умолчанию `1e-8`.
    pub fn new(
        parameters: Vec<Tensor>,
        lr: f32,
        betas: Option<(f32, f32)>,
        epsilon: Option<f32>,
    ) -> Self {
        let (beta1, beta2) = betas.unwrap_or((0.9, 0.999));
        let epsilon_val = epsilon.unwrap_or(1e-8);

        // Инициализируем буферы m и v нулями, с той же формой, что и у параметров.
        let mut m = Vec::with_capacity(parameters.len());
        let mut v = Vec::with_capacity(parameters.len());
        for p in &parameters {
            let shape = p.data.borrow().shape().to_vec();
            m.push(Tensor::zeros(&shape, false));
            v.push(Tensor::zeros(&shape, false));
        }

        Self {
            parameters,
            lr,
            beta1,
            beta2,
            epsilon: epsilon_val,
            t: 0,
            m,
            v,
        }
    }
}

impl Optimizer for Adam {
    /// Выполняет один шаг оптимизации Adam.
    fn step(&mut self) {
        // Увеличиваем счетчик шагов
        self.t += 1;

        // Проходим по всем параметрам и их соответствующим буферам m и v
        for ((p, m_p), v_p) in self.parameters.iter()
            .zip(self.m.iter_mut())
            .zip(self.v.iter_mut()) 
        {
            if let Some(grad_tensor) = &p.grad {
                let grad = grad_tensor.borrow();
                
                // --- Обновление моментов ---
                let mut m_p_data = m_p.data.borrow_mut();
                let mut v_p_data = v_p.data.borrow_mut();
                
                // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                *m_p_data = &*m_p_data * self.beta1 + &*grad * (1.0 - self.beta1);
                
                // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                let grad_sq = &*grad * &*grad;
                *v_p_data = &*v_p_data * self.beta2 + &grad_sq * (1.0 - self.beta2);
                
                // --- Коррекция смещения (Bias correction) ---
                let m_hat = &*m_p_data / (1.0 - self.beta1.powi(self.t));
                let v_hat = &*v_p_data / (1.0 - self.beta2.powi(self.t));
                
                // --- Обновление параметра ---
                // p_t = p_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)
                let v_hat_sqrt = v_hat.mapv(f32::sqrt);
                let update = &m_hat / &(v_hat_sqrt + self.epsilon);
                
                let mut p_data = p.data.borrow_mut();
                *p_data -= &(&update * self.lr);
            }
        }
    }

    /// Обнуляет градиенты для всех отслеживаемых параметров.
    fn zero_grad(&self) {
        for p in &self.parameters {
            if let Some(grad) = &p.grad {
                grad.borrow_mut().fill(0.0);
            }
        }
    }
}