use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use std::rc::Rc;

/// Транспонирует тензор, меняя местами две заданные оси.
pub fn transpose_op(a: &Tensor, axis1: usize, axis2: usize) -> Tensor {
    let mut permute_axes = (0..a.data.borrow().ndim()).collect::<Vec<_>>();
    permute_axes.swap(axis1, axis2);
    
    // Прямой проход
    // --- ИСПРАВЛЕНИЕ E0507: Клонируем данные, чтобы получить владение ---
    let result_data = a.data.borrow().clone().permuted_axes(permute_axes.clone());
    let mut result = Tensor::new(result_data, a.grad.is_some());

    // Обратный проход
    if a.grad.is_some() {
        let a_for_ctx = a.clone();
        let a_for_closure = a.clone();
        
        let backward_fn = Box::new(move |upstream_grad: &ArrayD<f32>| {
            if let Some(grad_a) = &a_for_closure.grad {
                // Здесь .view() корректен, так как permuted_axes может работать со ссылками
                let grad_update = upstream_grad.view().permuted_axes(permute_axes.clone());
                grad_a.borrow_mut().scaled_add(1.0, &grad_update.into_dyn());
            }
        });
        
        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_ctx],
            backward_fn,
        }));
    }

    result
}

/// Изменяет форму тензора без изменения данных.
pub fn reshape_op(a: &Tensor, new_shape: Vec<usize>) -> Tensor {
    // Прямой проход
    let result_data = a
        .data
        .borrow()
        .to_shape(IxDyn(&new_shape))
        .unwrap()
        .to_owned();
    let mut result = Tensor::new(result_data, a.grad.is_some());
    
    // Обратный проход
    if a.grad.is_some() {
        let a_for_ctx = a.clone();
        let a_for_closure = a.clone();
        let original_shape = a.data.borrow().shape().to_vec();
        
        let backward_fn = Box::new(move |upstream_grad: &ArrayD<f32>| {
            if let Some(grad_a) = &a_for_closure.grad {
                let grad_update = upstream_grad
                    .to_shape(IxDyn(&original_shape))
                    .unwrap()
                    .to_owned();
                grad_a.borrow_mut().scaled_add(1.0, &grad_update);
            }
        });
        
        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_ctx],
            backward_fn,
        }));
    }

    result
}