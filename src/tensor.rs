//! Модуль, определяющий `Tensor` — основную структуру данных в библиотеке.

use crate::core::autograd::{self, BackwardContext};
use crate::error::Result;
use ndarray::{ArrayD, IxDyn};
use std::cell::RefCell;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

/// Основная многомерная структура данных для всех операций.
///
/// `Tensor` является оберткой над `ndarray::ArrayD<f32>`, добавляющей
/// возможность автоматического вычисления градиентов (autograd).
///
/// Внутренние данные (`data`) и градиент (`grad`) обернуты в `Rc<RefCell<...>>`,
/// что позволяет иметь несколько "владельцев" одного тензора и изменять его
/// содержимое, даже если на него есть только иммутабельные ссылки. Это ключевой
/// механизм для построения динамического графа вычислений.
///
/// # Примеры
///
/// ```
/// use rusty_gradients::tensor::Tensor;
/// use ndarray::array;
///
/// // Создание тензора, требующего градиент
/// let data = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
/// let t = Tensor::new(data, true);
///
/// // Проверка, что градиент был инициализирован нулями
/// assert!(t.grad.is_some());
/// assert_eq!(t.grad.as_ref().unwrap().borrow().sum(), 0.0);
/// ```
#[derive(Clone)]
pub struct Tensor {
    /// Внутренние данные тензора. Доступны для чтения и записи через `RefCell`.
    pub data: Rc<RefCell<ArrayD<f32>>>,
    /// Градиент этого тензора. `None`, если `requires_grad` было `false`.
    pub grad: Option<Rc<RefCell<ArrayD<f32>>>>,
    /// Контекст для обратного распространения ошибки. `None` для "листовых" тензоров.
    pub ctx: Option<Rc<BackwardContext>>,
}

impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.data).hash(state);
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        // Сравниваем указатели, а не значения. Два тензора равны, только если
        // они являются одним и тем же узлом в графе вычислений.
        Rc::ptr_eq(&self.data, &other.data)
    }
}

impl Eq for Tensor {}

impl Tensor {
    /// Создает новый `Tensor`.
    ///
    /// # Аргументы
    ///
    /// * `data` - `ndarray::ArrayD<f32>`, который будет храниться в тензоре.
    /// * `requires_grad` - Если `true`, для этого тензора будет создан и
    ///   будет накапливаться градиент при обратном распространении ошибки.
    pub fn new(data: ArrayD<f32>, requires_grad: bool) -> Self {
        let grad = if requires_grad {
            let shape = data.shape();
            let grad_data = ArrayD::zeros(IxDyn(shape));
            Some(Rc::new(RefCell::new(grad_data)))
        } else {
            None
        };

        Self {
            data: Rc::new(RefCell::new(data)),
            grad,
            ctx: None,
        }
    }

    /// Создает новый `Tensor`, заполненный нулями.
    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        let data = ArrayD::zeros(IxDyn(shape));
        Self::new(data, requires_grad)
    }

    /// Создает новый `Tensor`, заполненный единицами.
    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        let data = ArrayD::ones(IxDyn(shape));
        Self::new(data, requires_grad)
    }

    /// Выполняет матричное умножение. См. `ops::matmul::dot_op`.
    pub fn dot(&self, other: &Tensor) -> Result<Tensor> {
        crate::ops::matmul::dot_op(self, other)
    }

    /// Выполняет вычитание. См. `ops::basic_ops`.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        self - other
    }

    /// Возводит каждый элемент тензора в степень. См. `ops::elementwise::powf_op`.
    pub fn powf(&self, power: f32) -> Tensor {
        crate::ops::elementwise::powf_op(self, power)
    }

    /// Суммирует все элементы тензора, возвращая скалярный тензор. См. `ops::reduction::sum_op`.
    pub fn sum(&self) -> Tensor {
        crate::ops::reduction::sum_op(self)
    }
    
    /// Применяет активацию ReLU. См. `ops::elementwise::relu_op`.
    pub fn relu(&self) -> Tensor {
        crate::ops::elementwise::relu_op(self)
    }
    
    /// Применяет активацию Sigmoid. См. `ops::elementwise::sigmoid_op`.
    pub fn sigmoid(&self) -> Tensor {
        crate::ops::elementwise::sigmoid_op(self)
    }

    /// Вычисляет натуральный логарифм каждого элемента. См. `ops::elementwise::log_op`.
    pub fn log(&self) -> Tensor {
        crate::ops::elementwise::log_op(self)
    }
    
    /// Выполняет операцию встраивания. См. `ops::embedding::embedding_op`.
    pub fn embedding(&self, weights: &Tensor) -> Result<Tensor> {
        crate::ops::embedding::embedding_op(self, weights)
    }

    /// Применяет Layer Normalization. См. `ops::norm::layernorm_op`.
    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, epsilon: f32) -> Result<Tensor> {
        crate::ops::norm::layernorm_op(self, gamma, beta, epsilon)
    }
    
    /// Применяет Softmax по последней оси. См. `ops::elementwise::softmax_op`.
    pub fn softmax(&self) -> Tensor {
        crate::ops::elementwise::softmax_op(self)
    }

    /// Транспонирует тензор. См. `ops::transform::transpose_op`.
    pub fn transpose(&self, axis1: usize, axis2: usize) -> Result<Tensor> {
        crate::ops::transform::transpose_op(self, axis1, axis2)
    }

    /// Изменяет форму тензора. См. `ops::transform::reshape_op`.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor> {
        crate::ops::transform::reshape_op(self, new_shape)
    }

    /// Запускает обратное распространение ошибки, начиная с этого тензора.
    ///
    /// Градиент самого тензора (`self.grad`) будет инициализирован единицами,
    /// после чего градиенты будут рекурсивно вычислены для всех его "предков"
    /// в графе вычислений, у которых `requires_grad` было `true`.
    pub fn backward(&self) {
        autograd::backward(self);
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.data.borrow();
        let grad_str = if let Some(grad) = &self.grad {
            format!("\n  grad: \n{}", grad.borrow())
        } else {
            "  grad: None".to_string()
        };

        write!(
            f,
            "Tensor {{\n  shape: {:?},\n  data: \n{},\n{}\n  ctx: {:?}\n}}",
            data.shape(),
            data,
            grad_str,
            self.ctx
        )
    }
}