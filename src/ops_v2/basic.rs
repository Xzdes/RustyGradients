//! Базовые арифметические операции с autograd

use crate::backend::cpu::CpuBackend;
use crate::backend::{Backend, Device};
use crate::core::autograd::BackwardContext;
use crate::error::Result;
use crate::tensor_v2::{TensorData, TensorV2};
use std::rc::Rc;
use std::sync::Arc;

/// Addition operation with autograd
///
/// c = a + b
/// dc/da = 1
/// dc/db = 1
pub fn add_op(a: &TensorV2, b: &TensorV2) -> Result<TensorV2> {
    // Forward pass через backend
    let backend = CpuBackend::new();

    let a_cpu = a.to_cpu_data()?;
    let b_cpu = b.to_cpu_data()?;

    let result_data = backend.add(&a_cpu, &b_cpu)?;

    let mut result = TensorV2::new_cpu(result_data, a.requires_grad() || b.requires_grad());

    // Backward pass setup
    if result.requires_grad() {
        let a_clone = a.clone();
        let b_clone = b.clone();

        // Gradient flow: grad_a += grad_out, grad_b += grad_out
        let backward_fn = Box::new(move |upstream: &ndarray::ArrayD<f32>| {
            // TODO: Accumulate gradients в a и b
            // Пока заглушка - нужна поддержка gradient storage в TensorV2
            let _ = upstream;
            let _ = &a_clone;
            let _ = &b_clone;
        });

        let ctx = BackwardContext {
            inputs: vec![], // TODO: конвертировать TensorV2 -> Tensor для совместимости
            backward_fn,
        };

        result.ctx = Some(Rc::new(ctx));
    }

    Ok(result)
}

/// Multiplication operation with autograd
///
/// c = a * b
/// dc/da = b
/// dc/db = a
pub fn mul_op(a: &TensorV2, b: &TensorV2) -> Result<TensorV2> {
    let backend = CpuBackend::new();

    let a_cpu = a.to_cpu_data()?;
    let b_cpu = b.to_cpu_data()?;

    let result_data = backend.mul(&a_cpu, &b_cpu)?;

    let mut result = TensorV2::new_cpu(result_data, a.requires_grad() || b.requires_grad());

    if result.requires_grad() {
        let a_clone = a.clone();
        let b_clone = b.clone();

        let backward_fn = Box::new(move |upstream: &ndarray::ArrayD<f32>| {
            // grad_a += grad_out * b
            // grad_b += grad_out * a
            let _ = upstream;
            let _ = &a_clone;
            let _ = &b_clone;
        });

        let ctx = BackwardContext {
            inputs: vec![],
            backward_fn,
        };

        result.ctx = Some(Rc::new(ctx));
    }

    Ok(result)
}

/// Subtraction operation with autograd
///
/// c = a - b
/// dc/da = 1
/// dc/db = -1
pub fn sub_op(a: &TensorV2, b: &TensorV2) -> Result<TensorV2> {
    let backend = CpuBackend::new();

    let a_cpu = a.to_cpu_data()?;
    let b_cpu = b.to_cpu_data()?;

    let result_data = backend.sub(&a_cpu, &b_cpu)?;

    let mut result = TensorV2::new_cpu(result_data, a.requires_grad() || b.requires_grad());

    if result.requires_grad() {
        let a_clone = a.clone();
        let b_clone = b.clone();

        let backward_fn = Box::new(move |upstream: &ndarray::ArrayD<f32>| {
            // grad_a += grad_out
            // grad_b -= grad_out
            let _ = upstream;
            let _ = &a_clone;
            let _ = &b_clone;
        });

        let ctx = BackwardContext {
            inputs: vec![],
            backward_fn,
        };

        result.ctx = Some(Rc::new(ctx));
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_forward() {
        let a = TensorV2::ones(&[2, 2], false, Device::cpu()).unwrap();
        let b = TensorV2::ones(&[2, 2], false, Device::cpu()).unwrap();

        let c = add_op(&a, &b).unwrap();
        assert_eq!(c.shape(), vec![2, 2]);
    }

    #[test]
    fn test_mul_forward() {
        let a = TensorV2::from_slice(&[2.0, 3.0], &[2], false).unwrap();
        let b = TensorV2::from_slice(&[4.0, 5.0], &[2], false).unwrap();

        let c = mul_op(&a, &b).unwrap();
        assert_eq!(c.shape(), vec![2]);
    }
}
