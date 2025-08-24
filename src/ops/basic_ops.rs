// src/ops/basic_ops.rs
// Add, Sub, Mul с корректным broadcasting и backward

use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
use ndarray::{Axis, ArrayD};
use std::ops::{Add, Sub, Mul};
use std::rc::Rc;

/// Сводит градиент `upstream` к форме `target_shape`
/// согласно правилам NumPy broadcasting (правые оси).
fn reduce_grad(upstream: &ArrayD<f32>, target: &[usize]) -> ArrayD<f32> {
    // Клонируем, чтобы не было заимствований
    let mut out = upstream.clone();

    // Проходим с конца (правые оси)
    for axis in (0..out.ndim()).rev() {
        // Если ось выходит за пределы target — сводим полностью
        if axis >= target.len() {
            out = out.sum_axis(Axis(axis));
            continue;
        }
        let tgt_len = target[axis];
        let cur_len = out.shape()[axis];
        if cur_len != tgt_len {
            if tgt_len == 1 {
                // Broadcasting: сводим
                out = out.sum_axis(Axis(axis));
            } else {
                panic!(
                    "reduce_grad: incompatible shapes {:?} -> {:?}",
                    upstream.shape(),
                    target
                );
            }
        }
    }

    // Финальный reshape
    if out.shape() != target {
        out = out
            .into_shape_with_order(target.to_vec())
            .expect("reduce_grad: final reshape failed");
    }
    out
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
            let lhs_shape = self.data.borrow().shape().to_vec();
            let rhs_shape = rhs.data.borrow().shape().to_vec();

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
            let lhs_shape = self.data.borrow().shape().to_vec();
            let rhs_shape = rhs.data.borrow().shape().to_vec();

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
            let lhs_shape = self.data.borrow().shape().to_vec();
            let rhs_shape = rhs.data.borrow().shape().to_vec();

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