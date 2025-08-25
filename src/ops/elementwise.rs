// src/ops/elementwise.rs
// Реализация поэлементных операций (ReLU, Sigmoid, Log, Powf, Softmax)
// с автоматическим дифференцированием через BackwardContext
// Поддерживаются тензоры любой размерности (2-D, 3-D, 4-D и выше)

use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
use ndarray::{Axis};
use std::rc::Rc;

/// powf: возводит каждый элемент тензора в степень `power`.
pub fn powf_op(a: &Tensor, power: f32) -> Tensor {
    // прямой проход
    let result_data = a.data.borrow().mapv(|val| val.powf(power));
    let requires_grad = a.grad.is_some();
    let mut result = Tensor::new(result_data, requires_grad);

    // обратный проход
    if requires_grad {
        // клонируем дважды: один клон для замыкания, второй для списка inputs
        let a_for_closure = a.clone();
        let a_for_inputs = a.clone();

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            let a_data = a_for_closure.data.borrow();
            let derivative = a_data.mapv(|val| power * val.powf(power - 1.0));
            if let Some(grad_a) = &a_for_closure.grad {
                grad_a.borrow_mut().scaled_add(1.0, &(upstream_grad * &derivative));
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_inputs],
            backward_fn,
        }));
    }

    result
}

/// ReLU: max(0, x)
pub fn relu_op(a: &Tensor) -> Tensor {
    let result_data = a.data.borrow().mapv(|val| val.max(0.0));
    let requires_grad = a.grad.is_some();
    let mut result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        let a_for_closure = a.clone();
        let a_for_inputs = a.clone();

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            let a_data = a_for_closure.data.borrow();
            let derivative = a_data.mapv(|val| if val > 0.0 { 1.0 } else { 0.0 });
            if let Some(grad_a) = &a_for_closure.grad {
                grad_a.borrow_mut().scaled_add(1.0, &(upstream_grad * &derivative));
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_inputs],
            backward_fn,
        }));
    }

    result
}

/// Sigmoid: 1 / (1 + e^(-x))
pub fn sigmoid_op(a: &Tensor) -> Tensor {
    let result_data = a.data.borrow().mapv(|val| 1.0 / (1.0 + (-val).exp()));
    let requires_grad = a.grad.is_some();
    let mut result = Tensor::new(result_data.clone(), requires_grad);

    if requires_grad {
        let a_for_closure = a.clone();
        let a_for_inputs = a.clone();
        let result_for_closure = result.clone();

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            let y = result_for_closure.data.borrow();
            let derivative = &*y * &(1.0 - &*y);
            if let Some(grad_a) = &a_for_closure.grad {
                grad_a.borrow_mut().scaled_add(1.0, &(upstream_grad * &derivative));
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_inputs],
            backward_fn,
        }));
    }

    result
}

/// Log: ln(x + ε)
pub fn log_op(a: &Tensor) -> Tensor {
    const EPS: f32 = 1e-8;
    let result_data = a.data.borrow().mapv(|val| (val + EPS).ln());
    let requires_grad = a.grad.is_some();
    let mut result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        let a_for_closure = a.clone();
        let a_for_inputs = a.clone();

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            let a_data = a_for_closure.data.borrow();
            let derivative = a_data.mapv(|val| 1.0 / (val + EPS));
            if let Some(grad_a) = &a_for_closure.grad {
                grad_a.borrow_mut().scaled_add(1.0, &(upstream_grad * &derivative));
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_inputs],
            backward_fn,
        }));
    }

    result
}

/// Softmax по последней оси (поддерживает любую размерность тензора).
pub fn softmax_op(x: &Tensor) -> Tensor {
    let x_data = x.data.borrow();
    let last_axis = x_data.ndim() - 1;

    // стабилизация: вычитаем максимум по последней оси
    let max_vals = x_data.map_axis(Axis(last_axis), |row| {
        row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v))
    });

    let mut result = x_data.clone();
    result
        .lanes_mut(Axis(last_axis))
        .into_iter()
        .zip(max_vals.iter())
        .for_each(|(mut lane, &max)| {
            lane.mapv_inplace(|v| (v - max).exp());
        });

    let sums = result.map_axis(Axis(last_axis), |row| row.sum());
    result
        .lanes_mut(Axis(last_axis))
        .into_iter()
        .zip(sums.iter())
        .for_each(|(mut lane, &sum)| {
            lane.mapv_inplace(|v| v / sum);
        });

    let requires_grad = x.grad.is_some();
    let mut out = Tensor::new(result, requires_grad);

    if requires_grad {
        let x_for_closure = x.clone();
        let x_for_inputs = x.clone();
        let out_for_closure = out.clone();

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            let y = out_for_closure.data.borrow();
            let last_axis = y.ndim() - 1;

            // grad = y * (dy - sum(dy * y))
            let sum_dy_y = (upstream_grad * &*y).map_axis(Axis(last_axis), |row| row.sum());
            let mut grad = y.clone();
            grad.lanes_mut(Axis(last_axis))
                .into_iter()
                .zip(sum_dy_y.iter())
                .for_each(|(mut lane, &s)| {
                    lane.mapv_inplace(|yi| yi * (yi - s));
                });

            if let Some(grad_x) = &x_for_closure.grad {
                grad_x.borrow_mut().scaled_add(1.0, &grad);
            }
        });

        out.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![x_for_inputs],
            backward_fn,
        }));
    }

    out
}