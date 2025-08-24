use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
use std::rc::Rc;

// Реализация операции возведения в степень.
pub fn powf_op(a: &Tensor, power: f32) -> Tensor {
    let result_data = a.data.borrow().mapv(|val| val.powf(power));
    let mut result = Tensor::new(result_data, a.grad.is_some());

    if a.grad.is_some() {
        // --- ИСПРАВЛЕНИЕ: Создаем два клона для контекста и замыкания ---
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

// Реализация операции ReLU (Rectified Linear Unit).
pub fn relu_op(a: &Tensor) -> Tensor {
    let result_data = a.data.borrow().mapv(|val| val.max(0.0));
    let mut result = Tensor::new(result_data, a.grad.is_some());

    if a.grad.is_some() {
        // --- ИСПРАВЛЕНИЕ: Создаем два клона для контекста и замыкания ---
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

// Реализация операции Sigmoid.
pub fn sigmoid_op(a: &Tensor) -> Tensor {
    let result_data = a.data.borrow().mapv(|val| 1.0 / (1.0 + (-val).exp()));
    let mut result = Tensor::new(result_data, a.grad.is_some());

    if a.grad.is_some() {
        // --- ИСПРАВЛЕНИЕ: Создаем два клона для контекста и замыкания ---
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