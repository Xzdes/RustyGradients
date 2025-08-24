// src/ops/basic_ops.rs
// Add, Sub, Mul с корректным broadcasting и backward
// Без E0506

use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
use ndarray::{Axis, ArrayD};
use std::ops::{Add, Sub, Mul};
use std::rc::Rc;

/// Сводит градиент `upstream` к `target_shape` по правилам NumPy broadcasting.
/// Суммируем по осям, где `target_shape == 1`.
fn reduce_grad(upstream: &ArrayD<f32>, target: &[usize]) -> ArrayD<f32> {
    let mut current = upstream.clone();
    let tgt_len = target.len();
    let cur_len = current.ndim();

    // Сводим лишние оси (слева)
    for axis in 0..(cur_len - tgt_len) {
        current = current.sum_axis(Axis(0));
    }

    // Теперь обрабатываем оставшиеся оси
    let cur_shape = current.shape().to_vec();
    for (axis, (&cur_len_axis, &tgt_len_axis)) in cur_shape
        .iter()
        .zip(target.iter())
        .enumerate()
    {
        if cur_len_axis != tgt_len_axis {
            if tgt_len_axis == 1 {
                current = current.sum_axis(Axis(axis));
            } else {
                panic!("reduce_grad: incompatible shapes {:?} -> {:?}", cur_shape, target);
            }
        }
    }

    // Финальный reshape
    if current.shape() != target {
        current = current
            .into_shape_with_order(target.to_vec())
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