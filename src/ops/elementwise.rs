use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
use ndarray::{Axis, Ix2}; // <-- Убираем `Order`
use std::rc::Rc;

// ... powf_op, relu_op, sigmoid_op, log_op остаются без изменений ...

pub fn powf_op(a: &Tensor, power: f32) -> Tensor {
    let result_data = a.data.borrow().mapv(|val| val.powf(power));
    let mut result = Tensor::new(result_data, a.grad.is_some());

    if a.grad.is_some() {
        let a_for_ctx = a.clone();
        let a_for_closure = a.clone();

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            let a_data = a_for_closure.data.borrow();
            let derivative = a_data.mapv(|val| power * val.powf(power - 1.0));
            if let Some(grad_a) = &a_for_closure.grad {
                *grad_a.borrow_mut() += &(upstream_grad * &derivative);
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_ctx],
            backward_fn,
        }));
    }
    result
}

pub fn relu_op(a: &Tensor) -> Tensor {
    let result_data = a.data.borrow().mapv(|val| val.max(0.0));
    let mut result = Tensor::new(result_data, a.grad.is_some());

    if a.grad.is_some() {
        let a_for_ctx = a.clone();
        let a_for_closure = a.clone();

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            let a_data = a_for_closure.data.borrow();
            let derivative = a_data.mapv(|val| if val > 0.0 { 1.0 } else { 0.0 });
            if let Some(grad_a) = &a_for_closure.grad {
                *grad_a.borrow_mut() += &(upstream_grad * &derivative);
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_ctx],
            backward_fn,
        }));
    }
    result
}

pub fn sigmoid_op(a: &Tensor) -> Tensor {
    let result_data = a.data.borrow().mapv(|val| 1.0 / (1.0 + (-val).exp()));
    let mut result = Tensor::new(result_data, a.grad.is_some());

    if a.grad.is_some() {
        let a_for_ctx = a.clone();
        let a_for_closure = a.clone();
        let result_for_closure = result.clone();

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            let result_data = result_for_closure.data.borrow();
            let derivative = &*result_data * &(1.0 - &*result_data);
            if let Some(grad_a) = &a_for_closure.grad {
                *grad_a.borrow_mut() += &(upstream_grad * &derivative);
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_ctx],
            backward_fn,
        }));
    }
    result
}

pub fn log_op(a: &Tensor) -> Tensor {
    const EPSILON: f32 = 1e-8;
    let result_data = a.data.borrow().mapv(|val| (val + EPSILON).ln());
    let mut result = Tensor::new(result_data, a.grad.is_some());

    if a.grad.is_some() {
        let a_for_ctx = a.clone();
        let a_for_closure = a.clone();

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            let a_data = a_for_closure.data.borrow();
            let derivative = 1.0 / (&*a_data + EPSILON);
            if let Some(grad_a) = &a_for_closure.grad {
                *grad_a.borrow_mut() += &(upstream_grad * &derivative);
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_ctx],
            backward_fn,
        }));
    }
    result
}


// --- ИСПРАВЛЕННАЯ ФУНКЦИЯ: Softmax ---
pub fn softmax_op(a: &Tensor) -> Tensor {
    let a_data = a.data.borrow();
    let a_2d = a_data.view().into_dimensionality::<Ix2>().unwrap();

    let max_vals = a_2d
        .map_axis(Axis(1), |row| {
            row.iter().fold(f32::NEG_INFINITY, |max, &v| v.max(max))
        })
        // --- ИСПРАВЛЕНИЕ: Убираем второй аргумент ---
        .into_shape_with_order((a_2d.nrows(), 1))
        .unwrap();
    let stabilized = &a_2d - &max_vals;
    
    let exponents = stabilized.mapv(f32::exp);
    
    // --- ИСПРАВЛЕНИЕ: Убираем второй аргумент ---
    let sum_exponents = exponents
        .sum_axis(Axis(1))
        .into_shape_with_order((exponents.nrows(), 1))
        .unwrap();
        
    let result_data = (&exponents / &sum_exponents).into_dyn();
    let mut result = Tensor::new(result_data, a.grad.is_some());

    if a.grad.is_some() {
        // --- ИСПРАВЛЕНИЕ: Паттерн с двумя клонами ---
        let a_for_ctx = a.clone();
        let a_for_closure = a.clone();
        let result_for_closure = result.clone();

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            let y = result_for_closure.data.borrow();
            let dy = upstream_grad;
            
            let dy_y = dy * &*y;
            let sum_dy_y = dy_y.sum_axis(Axis(1)).into_dyn();
            // --- ИСПРАВЛЕНИЕ: Убираем второй аргумент ---
            let sum_dy_y_reshaped = sum_dy_y
                .into_shape_with_order((dy.shape()[0], 1))
                .unwrap();
            
            let grad_a_update = &*y * (dy - &sum_dy_y_reshaped);

            if let Some(grad_a) = &a_for_closure.grad {
                grad_a.borrow_mut().scaled_add(1.0, &grad_a_update.into_dyn());
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_ctx],
            backward_fn,
        }));
    }

    result
}