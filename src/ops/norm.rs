use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
// --- ИСПРАВЛЕНИЕ: Удаляем Order, он нам не нужен в импортах ---
use ndarray::{ArrayD, Axis, Ix2};
use std::rc::Rc;

/// Выполняет операцию Layer Normalization.
pub fn layernorm_op(x: &Tensor, gamma: &Tensor, beta: &Tensor, epsilon: f32) -> Tensor {
    let x_data = x.data.borrow();
    let gamma_data = gamma.data.borrow();
    let beta_data = beta.data.borrow();

    // --- Прямой проход ---
    let mean = x_data.mean_axis(Axis(1)).unwrap();
    // --- ИСПРАВЛЕНИЕ: Убираем второй аргумент `Order::C` ---
    let mean_reshaped = mean
        .into_shape_with_order((x_data.shape()[0], 1))
        .unwrap();

    let x_minus_mean = &*x_data - &mean_reshaped;

    let variance = x_minus_mean.mapv(|v| v.powi(2)).mean_axis(Axis(1)).unwrap();
    // --- ИСПРАВЛЕНИЕ: Убираем второй аргумент `Order::C` ---
    let variance_reshaped = variance
        .clone()
        .into_shape_with_order((x_data.shape()[0], 1))
        .unwrap();

    let std_dev = (variance_reshaped + epsilon).mapv(f32::sqrt);
    
    let x_normalized = &x_minus_mean / &std_dev;

    let result_data = &x_normalized * &*gamma_data + &*beta_data;
    
    let requires_grad = x.grad.is_some() || gamma.grad.is_some() || beta.grad.is_some();
    let mut result = Tensor::new(result_data.into_dyn(), requires_grad);

    // --- Обратный проход ---
    if requires_grad {
        let x_for_ctx = x.clone();
        let gamma_for_ctx = gamma.clone();
        let beta_for_ctx = beta.clone();
        
        let x_for_closure = x.clone();
        let gamma_for_closure = gamma.clone();
        let beta_for_closure = beta.clone();

        let x_norm_for_closure = Tensor::new(x_normalized.into_dyn(), false);
        let std_inv_for_closure = Tensor::new((1.0 / &std_dev).into_dyn(), false);
        
        let backward_fn = Box::new(move |upstream_grad: &ArrayD<f32>| {
            let n_features = x_for_closure.data.borrow().shape()[1] as f32;
            let upstream_grad_2d = upstream_grad.view().into_dimensionality::<Ix2>().unwrap();
            
            let gamma_data_closure = gamma_for_closure.data.borrow();
            let x_norm_data_closure = x_norm_for_closure.data.borrow();
            let std_inv_data_closure = std_inv_for_closure.data.borrow();
            let x_data_closure = x_for_closure.data.borrow();

            // Градиенты для gamma и beta
            if let Some(gamma_grad) = &gamma_for_closure.grad {
                let dgamma = (upstream_grad * &*x_norm_data_closure).sum_axis(Axis(0));
                gamma_grad.borrow_mut().scaled_add(1.0, &dgamma.into_dyn());
            }
            if let Some(beta_grad) = &beta_for_closure.grad {
                let dbeta = upstream_grad.sum_axis(Axis(0));
                beta_grad.borrow_mut().scaled_add(1.0, &dbeta.into_dyn());
            }

            // Градиент для x
            if let Some(x_grad) = &x_for_closure.grad {
                let d_x_norm = &upstream_grad_2d * &*gamma_data_closure;
                
                // --- ИСПРАВЛЕНИЕ: Убираем второй аргумент `Order::C` во всех местах ---
                let d_variance = (&d_x_norm * &*x_data_closure * -0.5 * &std_inv_data_closure.mapv(|v| v.powi(3)))
                    .sum_axis(Axis(1))
                    .into_shape_with_order((x_data_closure.shape()[0], 1))
                    .unwrap();
                
                let d_mean = (&d_x_norm * -1.0 * &*std_inv_data_closure).sum_axis(Axis(1)).into_shape_with_order((x_data_closure.shape()[0], 1)).unwrap() -
                            (2.0 * &d_variance / n_features) * &x_data_closure.sum_axis(Axis(1)).into_shape_with_order((x_data_closure.shape()[0], 1)).unwrap();
                
                let dx = &d_x_norm * &*std_inv_data_closure + (2.0 / n_features) * &*x_data_closure * &d_variance + (1.0 / n_features) * &d_mean;

                x_grad.borrow_mut().scaled_add(1.0, &dx.into_dyn());
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![x_for_ctx, gamma_for_ctx, beta_for_ctx],
            backward_fn,
        }));
    }

    result
}