use crate::ops;
use ndarray::{ArrayD, IxDyn};
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
// use std::ops::Sub; // Убираем неиспользуемый импорт
use std::rc::Rc;

// ... остальной код файла без изменений ...

pub struct BackwardContext {
    pub inputs: Vec<Tensor>,
    pub backward_fn: Box<dyn Fn(&ArrayD<f32>)>,
}

impl fmt::Debug for BackwardContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BackwardContext")
            .field("num_inputs", &self.inputs.len())
            .finish()
    }
}

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
        ops::dot_op(self, other)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        self - other
    }

    pub fn powf(&self, power: f32) -> Tensor {
        ops::powf_op(self, power)
    }

    pub fn sum(&self) -> Tensor {
        ops::sum_op(self)
    }

    pub fn backward(&self) {
        let mut sorted_graph = Vec::new();
        let mut visited = HashSet::new();

        fn build_graph(tensor: &Tensor, visited: &mut HashSet<Tensor>, sorted: &mut Vec<Tensor>) {
            if visited.contains(tensor) {
                return;
            }
            visited.insert(tensor.clone());

            if let Some(ctx) = &tensor.ctx {
                for input_tensor in &ctx.inputs {
                    build_graph(input_tensor, visited, sorted);
                }
            }
            sorted.push(tensor.clone());
        }

        build_graph(self, &mut visited, &mut sorted_graph);

        if let Some(grad) = &self.grad {
            grad.borrow_mut().fill(1.0);
        } else {
            panic!("backward() called on a tensor that does not require gradients");
        }

        for tensor in sorted_graph.iter().rev() {
            if let Some(ctx) = &tensor.ctx {
                let upstream_grad = tensor.grad.as_ref().unwrap().borrow();
                (ctx.backward_fn)(&upstream_grad);
            }
        }
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