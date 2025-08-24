use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
use ndarray::{s, stack, Array, ArrayD, Axis, Dimension, Ix2};
use std::rc::Rc;
use std::ops::AddAssign;

/// Выполняет матричное умножение с поддержкой батчей (Batch Matmul).
pub fn dot_op(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();
    
    let a_shape = a_data.shape();
    let b_shape = b_data.shape();

    // --- Прямой проход ---
    let result_data = if a_data.ndim() > 2 {
        // Случай с батчами
        let batch_dims = &a_shape[..a_shape.len() - 2];
        let num_batches = batch_dims.iter().product();

        let a_reshaped = a_data.to_shape((num_batches, a_shape[a_shape.len()-2], a_shape[a_shape.len()-1])).unwrap();
        let b_reshaped = b_data.to_shape((num_batches, b_shape[b_shape.len()-2], b_shape[b_shape.len()-1])).unwrap();

        let mut result_slices = Vec::with_capacity(num_batches);
        for i in 0..num_batches {
            let a_mat = a_reshaped.slice(s![i, .., ..]);
            let b_mat = b_reshaped.slice(s![i, .., ..]);
            result_slices.push(a_mat.dot(&b_mat));
        }
        
        let view: Vec<_> = result_slices.iter().map(|arr| arr.view()).collect();
        let mut final_shape = batch_dims.to_vec();
        final_shape.extend_from_slice(&[a_shape[a_shape.len()-2], b_shape[b_shape.len()-1]]);

        stack(Axis(0), &view).unwrap().into_shape(final_shape).unwrap().into_dyn()
    } else {
        // Стандартное 2D умножение
        a_data.view().into_dimensionality::<Ix2>().unwrap().dot(&b_data.view().into_dimensionality::<Ix2>().unwrap()).into_dyn()
    };

    let requires_grad = a.grad.is_some() || b.grad.is_some();
    let mut result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        let a_for_ctx = a.clone();
        let b_for_ctx = b.clone();
        let a_for_closure = a.clone();
        let b_for_closure = b.clone();

        let backward_fn = Box::new(move |upstream_grad: &ArrayD<f32>| {
            let a_closure_data = a_for_closure.data.borrow();
            let b_closure_data = b_for_closure.data.borrow();
            let a_shape = a_closure_data.shape();
            let b_shape = b_closure_data.shape();
            let up_shape = upstream_grad.shape();

            if a_closure_data.ndim() > 2 {
                let batch_dims = &a_shape[..a_shape.len() - 2];
                let num_batches = batch_dims.iter().product();

                let a_reshaped = a_closure_data.to_shape((num_batches, a_shape[a_shape.len()-2], a_shape[a_shape.len()-1])).unwrap();
                let b_reshaped = b_closure_data.to_shape((num_batches, b_shape[b_shape.len()-2], b_shape[b_shape.len()-1])).unwrap();
                let up_reshaped = upstream_grad.to_shape((num_batches, up_shape[up_shape.len()-2], up_shape[up_shape.len()-1])).unwrap();

                if let Some(grad_a) = &a_for_closure.grad {
                    let mut grad_a_full = grad_a.borrow_mut();
                    let mut grad_a_reshaped = grad_a_full.view_mut().into_shape((num_batches, a_shape[a_shape.len()-2], a_shape[a_shape.len()-1])).unwrap();

                    for i in 0..num_batches {
                        let b_mat = b_reshaped.slice(s![i, .., ..]);
                        let up_mat = up_reshaped.slice(s![i, .., ..]);
                        grad_a_reshaped.slice_mut(s![i, .., ..]).add_assign(&up_mat.dot(&b_mat.t()));
                    }
                }
                if let Some(grad_b) = &b_for_closure.grad {
                    let mut grad_b_full = grad_b.borrow_mut();
                    let mut grad_b_reshaped = grad_b_full.view_mut().into_shape((num_batches, b_shape[b_shape.len()-2], b_shape[b_shape.len()-1])).unwrap();

                    for i in 0..num_batches {
                        let a_mat = a_reshaped.slice(s![i, .., ..]);
                        let up_mat = up_reshaped.slice(s![i, .., ..]);
                        grad_b_reshaped.slice_mut(s![i, .., ..]).add_assign(&a_mat.t().dot(&up_mat));
                    }
                }
            } else {
                let a_view = a_closure_data.view().into_dimensionality::<Ix2>().unwrap();
                let b_view = b_closure_data.view().into_dimensionality::<Ix2>().unwrap();
                let upstream_view = upstream_grad.view().into_dimensionality::<Ix2>().unwrap();
                if let Some(grad_a) = &a_for_closure.grad {
                    grad_a.borrow_mut().scaled_add(1.0, &upstream_view.dot(&b_view.t()).into_dyn());
                }
                if let Some(grad_b) = &b_for_closure.grad {
                    grad_b.borrow_mut().scaled_add(1.0, &a_view.t().dot(&upstream_view).into_dyn());
                }
            }
        });
        
        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_ctx, b_for_ctx],
            backward_fn,
        }));
    }

    result
}