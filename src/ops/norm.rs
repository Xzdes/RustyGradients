//! Модуль, реализующий операцию Layer Normalization.

use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
// --- ИЗМЕНЕНИЕ: Импортируем типы ошибок ---
use crate::error::{Result, RustyGradientsError};
use ndarray::{ArrayD, Axis};
use std::rc::Rc;

/// Выполняет операцию Layer Normalization.
/// Эта версия является обобщенной и работает с тензорами любой размерности (2D, 3D, ...),
/// выполняя нормализацию вдоль последней оси.
// --- ИЗМЕНЕНИЕ: Сигнатура функции обновлена ---
pub fn layernorm_op(x: &Tensor, gamma: &Tensor, beta: &Tensor, epsilon: f32) -> Result<Tensor> {
    let x_data = x.data.borrow();
    let gamma_data = gamma.data.borrow();
    let beta_data = beta.data.borrow();

    let last_axis = x_data.ndim() - 1;
    let axis = Axis(last_axis);
    let n_features = x_data.shape()[last_axis] as f32;
    
    // --- ИЗМЕНЕНИЕ: Добавляем проверку форм для gamma и beta ---
    let expected_param_shape = [1, n_features as usize];
    if gamma_data.shape() != expected_param_shape {
        return Err(RustyGradientsError::ShapeError(format!(
            "Invalid shape for gamma in LayerNorm: expected {:?}, got {:?}",
            expected_param_shape, gamma_data.shape()
        )));
    }
    if beta_data.shape() != expected_param_shape {
        return Err(RustyGradientsError::ShapeError(format!(
            "Invalid shape for beta in LayerNorm: expected {:?}, got {:?}",
            expected_param_shape, beta_data.shape()
        )));
    }


    // --- Прямой проход ---

    let mean = x_data.mean_axis(axis).unwrap();
    let mean_reshaped = mean.insert_axis(axis);

    let x_minus_mean = &*x_data - &mean_reshaped;

    let variance = x_minus_mean.mapv(|v| v.powi(2)).mean_axis(axis).unwrap();
    let variance_reshaped = variance.insert_axis(axis);

    let std_dev_inv = (&variance_reshaped + epsilon).mapv(f32::sqrt).mapv(|v| 1.0 / v);
    let x_normalized = &x_minus_mean * &std_dev_inv;

    let result_data = &x_normalized * &*gamma_data + &*beta_data;

    let requires_grad = x.grad.is_some() || gamma.grad.is_some() || beta.grad.is_some();
    let mut result = Tensor::new(result_data, requires_grad);

    // --- Обратный проход ---
    if requires_grad {
        let x_for_inputs = x.clone();
        let gamma_for_inputs = gamma.clone();
        let beta_for_inputs = beta.clone();

        let x_for_closure = x.clone();
        let gamma_for_closure = gamma.clone();
        let beta_for_closure = beta.clone();
        
        let x_minus_mean_for_closure = Tensor::new(x_minus_mean, false);
        let x_norm_for_closure = Tensor::new(x_normalized, false);
        let std_dev_inv_for_closure = Tensor::new(std_dev_inv, false);

        let backward_fn = Box::new(move |upstream_grad: &ArrayD<f32>| {
            let gamma_data_closure = gamma_for_closure.data.borrow();
            let x_norm_data_closure = x_norm_for_closure.data.borrow();
            let x_minus_mean_data_closure = x_minus_mean_for_closure.data.borrow();
            let std_dev_inv_data_closure = std_dev_inv_for_closure.data.borrow();

            let reduction_axes: Vec<usize> = (0..upstream_grad.ndim() - 1).collect();

            if let Some(gamma_grad) = &gamma_for_closure.grad {
                let mut dgamma = upstream_grad * &*x_norm_data_closure;
                for &axis_idx in reduction_axes.iter().rev() {
                    dgamma = dgamma.sum_axis(Axis(axis_idx));
                }
                gamma_grad.borrow_mut().scaled_add(1.0, &dgamma.into_dyn());
            }
            if let Some(beta_grad) = &beta_for_closure.grad {
                let mut dbeta = upstream_grad.clone();
                for &axis_idx in reduction_axes.iter().rev() {
                    dbeta = dbeta.sum_axis(Axis(axis_idx));
                }
                beta_grad.borrow_mut().scaled_add(1.0, &dbeta.into_dyn());
            }

            if let Some(x_grad) = &x_for_closure.grad {
                let last_axis_bw = upstream_grad.ndim() - 1;
                let axis_bw = Axis(last_axis_bw);

                let d_x_norm = upstream_grad * &*gamma_data_closure;
                let d_std_dev_inv = (&d_x_norm * &*x_minus_mean_data_closure).sum_axis(axis_bw);
                let d_std_dev_inv_reshaped = d_std_dev_inv.insert_axis(axis_bw);
                let d_variance = &d_std_dev_inv_reshaped * -0.5 * &std_dev_inv_data_closure.mapv(|v| v.powi(3));
                let d_x_minus_mean_part1 = &d_x_norm * &*std_dev_inv_data_closure;
                let d_x_minus_mean_part2 = &d_variance * (2.0 / n_features) * &*x_minus_mean_data_closure;
                let d_x_minus_mean = &d_x_minus_mean_part1 + &d_x_minus_mean_part2;
                let d_mean = -d_x_minus_mean.sum_axis(axis_bw).insert_axis(axis_bw);
                let dx = d_x_minus_mean + (1.0 / n_features) * d_mean;
                x_grad.borrow_mut().scaled_add(1.0, &dx);
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![x_for_inputs, gamma_for_inputs, beta_for_inputs],
            backward_fn,
        }));
    }

    // --- ИЗМЕНЕНИЕ: Оборачиваем результат в Ok() ---
    Ok(result)
}