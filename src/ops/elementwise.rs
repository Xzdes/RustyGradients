// src/ops/elementwise.rs
use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
use ndarray::Axis;
use std::rc::Rc;

pub fn exp_op(a: &Tensor) -> Tensor {
    let result_data = a.data.borrow().mapv(|val| val.exp());
    let requires_grad = a.grad.is_some();
    let mut result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        // ИСПРАВЛЕНИЕ: Создаем две копии. Одна для замыкания, другая для inputs.
        let a_for_closure = a.clone();
        let a_for_inputs = a.clone();
        let result_for_closure = result.clone(); 

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            if let Some(grad_a) = &a_for_closure.grad { 
                let derivative = result_for_closure.data.borrow();
                grad_a.borrow_mut().scaled_add(1.0, &(upstream_grad * &*derivative));
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_inputs], 
            backward_fn,
        }));
    }

    result
}

pub fn powf_op(a: &Tensor, power: f32) -> Tensor {
    let result_data = a.data.borrow().mapv(|val| val.powf(power));
    let requires_grad = a.grad.is_some();
    let mut result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        // ИСПРАВЛЕНИЕ: Та же логика
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

pub fn relu_op(a: &Tensor) -> Tensor {
    let result_data = a.data.borrow().mapv(|val| val.max(0.0));
    let requires_grad = a.grad.is_some();
    let mut result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        // ИСПРАВЛЕНИЕ: Та же логика
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

pub fn sigmoid_op(a: &Tensor) -> Tensor {
    let result_data = a.data.borrow().mapv(|val| 1.0 / (1.0 + (-val).exp()));
    let requires_grad = a.grad.is_some();
    let mut result = Tensor::new(result_data.clone(), requires_grad);

    if requires_grad {
        // ИСПРАВЛЕНИЕ: Та же логика
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

pub fn log_op(a: &Tensor) -> Tensor {
    const EPS: f32 = 1e-8;
    let result_data = a.data.borrow().mapv(|val| (val + EPS).ln());
    let requires_grad = a.grad.is_some();
    let mut result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        // ИСПРАВЛЕНИЕ: Та же логика
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

pub fn softmax_op(x: &Tensor) -> Tensor {
    let x_data = x.data.borrow();
    let last_axis = Axis(x_data.ndim() - 1);

    let max_vals = x_data.map_axis(last_axis, |row| {
        row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v))
    });
    let max_vals_reshaped = max_vals.insert_axis(last_axis);

    let x_stabilized = &*x_data - &max_vals_reshaped;
    let x_exp = x_stabilized.mapv(|v| v.exp());
    
    let sums = x_exp.sum_axis(last_axis).insert_axis(last_axis);
    let result_data = &x_exp / &sums;

    let requires_grad = x.grad.is_some();
    let mut out = Tensor::new(result_data, requires_grad);

    if requires_grad {
        // ИСПРАВЛЕНИЕ: Та же логика
        let x_for_closure = x.clone();
        let x_for_inputs = x.clone();
        let out_for_closure = out.clone();

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            let y = out_for_closure.data.borrow();
            
            let sum_dy_y = (upstream_grad * &*y).sum_axis(Axis(y.ndim()-1)).insert_axis(Axis(y.ndim()-1));
            let grad = &*y * &(upstream_grad - &sum_dy_y);

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