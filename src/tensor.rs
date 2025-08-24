// --- ИСПРАВЛЕНИЕ: Удаляем `crate::ops` ---
use crate::core::autograd::{self, BackwardContext};
use ndarray::{ArrayD, IxDyn};
use std::cell::RefCell;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
// --- ИСПРАВЛЕНИЕ: Удаляем `std::ops::Sub`, так как он используется только в методе `sub`, а не во всем файле. ---

#[derive(Clone)]
pub struct Tensor {
    pub data: Rc<RefCell<ArrayD<f32>>>,
    pub grad: Option<Rc<RefCell<ArrayD<f32>>>>,
    pub ctx: Option<Rc<BackwardContext>>,
}

impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.data).hash(state);
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }
}

impl Eq for Tensor {}

impl Tensor {
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

    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        let data = ArrayD::zeros(IxDyn(shape));
        Self::new(data, requires_grad)
    }

    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        let data = ArrayD::ones(IxDyn(shape));
        Self::new(data, requires_grad)
    }

    pub fn dot(&self, other: &Tensor) -> Tensor {
        crate::ops::matmul::dot_op(self, other)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        // Чтобы использовать оператор `-`, трейт Sub должен быть в области видимости.
        // Мы можем импортировать его прямо здесь, локально для метода.
        use std::ops::Sub;
        self - other
    }

    pub fn powf(&self, power: f32) -> Tensor {
        crate::ops::elementwise::powf_op(self, power)
    }

    pub fn sum(&self) -> Tensor {
        crate::ops::reduction::sum_op(self)
    }

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