//! Модуль, реализующий операции трансформации тензоров, такие как reshape и transpose.

use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
// --- ИЗМЕНЕНИЕ: Импортируем типы ошибок ---
use crate::error::{Result, RustyGradientsError};
use ndarray::{ArrayD, IxDyn};
use std::rc::Rc;

/// Транспонирует тензор, меняя местами две заданные оси.
///
/// Эта операция не может завершиться ошибкой, если оси находятся в пределах размерности,
/// но для единообразия API она также возвращает `Result`.
pub fn transpose_op(a: &Tensor, axis1: usize, axis2: usize) -> Result<Tensor> {
    // Прямой проход
    let a_data = a.data.borrow();
    if axis1 >= a_data.ndim() || axis2 >= a_data.ndim() {
        return Err(RustyGradientsError::InvalidInput(format!(
            "Transpose axes ({}, {}) out of bounds for tensor with {} dimensions",
            axis1,
            axis2,
            a_data.ndim()
        )));
    }

    let mut permute_axes = (0..a_data.ndim()).collect::<Vec<_>>();
    permute_axes.swap(axis1, axis2);

    let result_data = a_data.clone().permuted_axes(permute_axes.clone());
    let mut result = Tensor::new(result_data, a.grad.is_some());

    // Обратный проход
    if a.grad.is_some() {
        let a_for_ctx = a.clone();
        let a_for_closure = a.clone();

        let backward_fn = Box::new(move |upstream_grad: &ArrayD<f32>| {
            if let Some(grad_a) = &a_for_closure.grad {
                let grad_update = upstream_grad.view().permuted_axes(permute_axes.clone());
                grad_a.borrow_mut().scaled_add(1.0, &grad_update.into_dyn());
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_ctx],
            backward_fn,
        }));
    }

    Ok(result)
}

/// Изменяет форму тензора без изменения данных.
///
/// Может вернуть ошибку `RustyGradientsError::ShapeError`, если общее
/// количество элементов в `new_shape` не совпадает с количеством элементов в тензоре `a`.
pub fn reshape_op(a: &Tensor, new_shape: Vec<usize>) -> Result<Tensor> {
    // Прямой проход
    // --- ИЗМЕНЕНИЕ: Заменяем `.unwrap()` на обработку ошибки ---
    let result_data = a
        .data
        .borrow()
        .to_shape(IxDyn(&new_shape))
        // `map_err` преобразует ошибку от `ndarray` в нашу кастомную ошибку
        .map_err(|e| RustyGradientsError::ShapeError(e.to_string()))?
        .to_owned();
    let mut result = Tensor::new(result_data, a.grad.is_some());

    // Обратный проход
    if a.grad.is_some() {
        let a_for_ctx = a.clone();
        let a_for_closure = a.clone();
        let original_shape = a.data.borrow().shape().to_vec();

        let backward_fn = Box::new(move |upstream_grad: &ArrayD<f32>| {
            if let Some(grad_a) = &a_for_closure.grad {
                // Здесь .unwrap() допустим, так как мы знаем, что original_shape корректна.
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

    Ok(result)
}