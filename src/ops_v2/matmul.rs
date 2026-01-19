//! Matrix multiplication with autograd

use crate::backend::cpu::CpuBackend;
use crate::backend::Backend;
use crate::core::autograd::BackwardContext;
use crate::error::Result;
use crate::tensor_v2::TensorV2;
use std::rc::Rc;

/// Matrix multiplication with autograd
///
/// c = a @ b
/// dc/da = grad_out @ b^T
/// dc/db = a^T @ grad_out
pub fn matmul_op(a: &TensorV2, b: &TensorV2) -> Result<TensorV2> {
    let backend = CpuBackend::new();

    let a_cpu = a.to_cpu_data()?;
    let b_cpu = b.to_cpu_data()?;

    let result_data = backend.matmul(&a_cpu, &b_cpu)?;

    let mut result = TensorV2::new_cpu(result_data, a.requires_grad() || b.requires_grad());

    if result.requires_grad() {
        let a_clone = a.clone();
        let b_clone = b.clone();

        let backward_fn = Box::new(move |upstream: &ndarray::ArrayD<f32>| {
            // For 2D matmul:
            // grad_a = grad_out @ b^T
            // grad_b = a^T @ grad_out

            // TODO: Полная реализация backward для matmul
            // Учитывая batched dimensions (3D, 4D)
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
    fn test_matmul_2d() {
        let a = TensorV2::randn(&[3, 4], false);
        let b = TensorV2::randn(&[4, 5], false);

        let c = matmul_op(&a, &b).unwrap();
        assert_eq!(c.shape(), vec![3, 5]);
    }

    #[test]
    fn test_matmul_batched() {
        let a = TensorV2::randn(&[2, 3, 4], false);
        let b = TensorV2::randn(&[2, 4, 5], false);

        let c = matmul_op(&a, &b).unwrap();
        assert_eq!(c.shape(), vec![2, 3, 5]);
    }
}
