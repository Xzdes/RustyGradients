use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
use ndarray::{ArrayD, Axis};
use std::ops::{Add, Mul, Sub};
use std::rc::Rc;

/// Сводит градиент `upstream` к `target_shape` по правилам NumPy broadcasting.
fn reduce_grad(upstream: &ArrayD<f32>, target_shape: &[usize]) -> ArrayD<f32> {
    let mut current = upstream.clone();
    let upstream_shape = current.shape();
    
    let trim_dims = upstream_shape.len().saturating_sub(target_shape.len());
    for _ in 0..trim_dims {
        current = current.sum_axis(Axis(0));
    }

    let mut axes_to_sum = Vec::new();
    for (i, (&upstream_dim, &target_dim)) in current.shape().iter().zip(target_shape.iter()).enumerate() {
        if upstream_dim != target_dim {
            if target_dim == 1 {
                axes_to_sum.push(i);
            } else {
                 panic!(
                    "Cannot reduce grad from shape {:?} to {:?}, dimension mismatch at axis {}",
                    upstream.shape(), target_shape, i
                );
            }
        }
    }
    
    for axis_idx in axes_to_sum.iter().rev() {
        current = current.sum_axis(Axis(*axis_idx));
    }

    if current.shape() != target_shape {
        current = current
            .into_shape_with_order(target_shape.to_vec())
            .expect("reduce_grad: final reshape failed");
    }

    current
}

// ------------------ Add ------------------
impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Self::Output {
        let lhs_data = self.data.borrow();
        let rhs_data = rhs.data.borrow();
        let result_data = &*lhs_data + &*rhs_data;
        let requires_grad = self.grad.is_some() || rhs.grad.is_some();
        let mut result = Tensor::new(result_data, requires_grad);

        if requires_grad {
            let lhs_shape = lhs_data.shape().to_vec();
            let rhs_shape = rhs_data.shape().to_vec();
            let lhs_for_closure = self.clone();
            let rhs_for_closure = rhs.clone();
            let backward_fn = Box::new(move |upstream_grad: &ArrayD<f32>| {
                if let Some(grad_lhs) = &lhs_for_closure.grad {
                    let reduced = reduce_grad(upstream_grad, &lhs_shape);
                    grad_lhs.borrow_mut().scaled_add(1.0, &reduced);
                }
                if let Some(grad_rhs) = &rhs_for_closure.grad {
                    let reduced = reduce_grad(upstream_grad, &rhs_shape);
                    grad_rhs.borrow_mut().scaled_add(1.0, &reduced);
                }
            });
            result.ctx = Some(Rc::new(BackwardContext {
                inputs: vec![self.clone(), rhs.clone()],
                backward_fn,
            }));
        }
        result
    }
}

// ------------------ Sub ------------------
impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        let lhs_data = self.data.borrow();
        let rhs_data = rhs.data.borrow();
        let result_data = &*lhs_data - &*rhs_data;
        let requires_grad = self.grad.is_some() || rhs.grad.is_some();
        let mut result = Tensor::new(result_data, requires_grad);

        if requires_grad {
            let lhs_shape = lhs_data.shape().to_vec();
            let rhs_shape = rhs_data.shape().to_vec();
            let lhs_for_closure = self.clone();
            let rhs_for_closure = rhs.clone();
            let backward_fn = Box::new(move |upstream_grad: &ArrayD<f32>| {
                if let Some(grad_lhs) = &lhs_for_closure.grad {
                    let reduced = reduce_grad(upstream_grad, &lhs_shape);
                    grad_lhs.borrow_mut().scaled_add(1.0, &reduced);
                }
                if let Some(grad_rhs) = &rhs_for_closure.grad {
                    let reduced = reduce_grad(upstream_grad, &rhs_shape);
                    grad_rhs.borrow_mut().scaled_add(-1.0, &reduced);
                }
            });
            result.ctx = Some(Rc::new(BackwardContext {
                inputs: vec![self.clone(), rhs.clone()],
                backward_fn,
            }));
        }
        result
    }
}

// ------------------ Mul ------------------
impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        let lhs_data = self.data.borrow();
        let rhs_data = rhs.data.borrow();
        let result_data = &*lhs_data * &*rhs_data;
        let requires_grad = self.grad.is_some() || rhs.grad.is_some();
        let mut result = Tensor::new(result_data, requires_grad);

        if requires_grad {
            let lhs_shape = lhs_data.shape().to_vec();
            let rhs_shape = rhs_data.shape().to_vec();
            let lhs_for_closure = self.clone();
            let rhs_for_closure = rhs.clone();
            let backward_fn = Box::new(move |upstream_grad: &ArrayD<f32>| {
                if let Some(grad_lhs) = &lhs_for_closure.grad {
                    let rhs_val = rhs_for_closure.data.borrow();
                    let reduced = reduce_grad(&(upstream_grad * &*rhs_val), &lhs_shape);
                    grad_lhs.borrow_mut().scaled_add(1.0, &reduced);
                }
                if let Some(grad_rhs) = &rhs_for_closure.grad {
                    let lhs_val = lhs_for_closure.data.borrow();
                    let reduced = reduce_grad(&(upstream_grad * &*lhs_val), &rhs_shape);
                    grad_rhs.borrow_mut().scaled_add(1.0, &reduced);
                }
            });
            result.ctx = Some(Rc::new(BackwardContext {
                inputs: vec![self.clone(), rhs.clone()],
                backward_fn,
            }));
        }
        result
    }
}