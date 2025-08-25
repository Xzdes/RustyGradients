//! Модуль, реализующий операцию встраивания (Embedding).

use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
use crate::error::{Result, RustyGradientsError};
use ndarray::{s, Array, Axis};
use std::rc::Rc;

/// Выполняет операцию Embedding (встраивания).
///
/// # Аргументы
///
/// * `ids` - Тензор с целочисленными индексами (ID токенов). Ожидается 2D-тензор.
/// * `weights` - Матрица весов Embedding-слоя. 2D-тензор формы `[vocab_size, embedding_dim]`.
///
/// # Возвращает
///
/// 3D-тензор формы `[batch_size, seq_len, embedding_dim]`.
///
/// # Ошибки
///
/// Вернет `RustyGradientsError::InvalidInput`, если какой-либо из индексов
/// в `ids` будет больше или равен `vocab_size`.
/// Вернет `RustyGradientsError::DimensionError`, если `ids` или `weights` не являются 2D-тензорами.
pub fn embedding_op(ids: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let ids_data = ids.data.borrow();
    let weights_data = weights.data.borrow();

    if ids_data.ndim() != 2 {
        return Err(RustyGradientsError::DimensionError {
            expected: 2,
            actual: ids_data.ndim(),
        });
    }
    if weights_data.ndim() != 2 {
        return Err(RustyGradientsError::DimensionError {
            expected: 2,
            actual: weights_data.ndim(),
        });
    }

    let batch_size = ids_data.shape()[0];
    let seq_len = ids_data.shape()[1];
    let vocab_size = weights_data.shape()[0];
    let embedding_dim = weights_data.shape()[1];

    let mut result_vec: Vec<f32> = Vec::with_capacity(batch_size * seq_len * embedding_dim);

    for id_val in ids_data.iter() {
        let id = *id_val as usize;
        if id >= vocab_size {
            return Err(RustyGradientsError::InvalidInput(format!(
                "Index {} out of bounds for vocabulary size {}",
                id, vocab_size
            )));
        }
        let embedding_vector = weights_data.index_axis(Axis(0), id);
        result_vec.extend_from_slice(embedding_vector.as_slice().unwrap());
    }

    let result_shape = vec![batch_size, seq_len, embedding_dim];
    let result_array = Array::from_shape_vec(result_shape, result_vec)
        .map_err(|e| RustyGradientsError::ShapeError(e.to_string()))?
        .into_dyn();
        
    let mut result = Tensor::new(result_array, weights.grad.is_some());

    if weights.grad.is_some() {
        let ids_for_ctx = ids.clone();
        let weights_for_ctx = weights.clone();
        let ids_for_closure = ids.clone();
        let weights_for_closure = weights.clone();

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            if let Some(weights_grad) = &weights_for_closure.grad {
                let mut weights_grad_borrow = weights_grad.borrow_mut();
                let ids_data_closure = ids_for_closure.data.borrow();

                for (i, id_val) in ids_data_closure.iter().enumerate() {
                    let id = *id_val as usize;
                    let grad_slice = upstream_grad.slice(s![i / seq_len, i % seq_len, ..]);
                    let mut weight_grad_row = weights_grad_borrow.index_axis_mut(Axis(0), id);
                    weight_grad_row += &grad_slice;
                }
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![ids_for_ctx, weights_for_ctx],
            backward_fn,
        }));
    }

    Ok(result)
}