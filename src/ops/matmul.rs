use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
use ndarray::{s, ArrayD, Ix2, Ix3, Ix4};
use std::ops::AddAssign;
use std::rc::Rc;

/// batched dot (matrix multiplication) для:
/// - 4-D ⊗ 4-D → 4-D
/// - 3-D ⊗ 2-D → 3-D
/// - 3-D ⊗ 3-D → 3-D
/// - 2-D ⊗ 2-D → 2-D
pub fn dot_op(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();

    let result_data: ArrayD<f32> = match (a_data.ndim(), b_data.ndim()) {
        // ---------- 4-D ⊗ 4-D ----------
        (4, 4) => {
            let batch_size = a_data.shape()[0];
            let heads = a_data.shape()[1];
            let m = a_data.shape()[2];
            let k = a_data.shape()[3];
            let n = b_data.shape()[3];

            assert_eq!(b_data.shape(), &[batch_size, heads, k, n]);

            let mut out = ArrayD::zeros(vec![batch_size, heads, m, n]);

            for b in 0..batch_size {
                for h in 0..heads {
                    let a_mat = a_data
                        .slice(s![b, h, .., ..])
                        .into_dimensionality::<Ix2>()
                        .unwrap();
                    let b_mat = b_data
                        .slice(s![b, h, .., ..])
                        .into_dimensionality::<Ix2>()
                        .unwrap();
                    let mut c_mat = out
                        .slice_mut(s![b, h, .., ..])
                        .into_dimensionality::<Ix2>()
                        .unwrap();
                    c_mat.assign(&a_mat.dot(&b_mat));
                }
            }

            out
        }

        // ---------- 3-D ⊗ 2-D ----------
        (3, 2) => {
            let batch = a_data.shape()[0];
            let seq = a_data.shape()[1];
            let in_dim = a_data.shape()[2];
            let out_dim = b_data.shape()[1];

            assert_eq!(b_data.shape(), &[in_dim, out_dim]);

            let mut out = ArrayD::zeros(vec![batch, seq, out_dim]);

            for b_idx in 0..batch {
                let a_mat = a_data
                    .slice(s![b_idx, .., ..])
                    .into_dimensionality::<Ix2>()
                    .unwrap();
                let b_mat = b_data.view().into_dimensionality::<Ix2>().unwrap();
                let mut out_mat = out
                    .slice_mut(s![b_idx, .., ..])
                    .into_dimensionality::<Ix2>()
                    .unwrap();
                out_mat.assign(&a_mat.dot(&b_mat));
            }

            out
        }

        // ---------- 3-D ⊗ 3-D ----------
        (3, 3) => {
            let batch = a_data.shape()[0];
            let m = a_data.shape()[1];
            let k = a_data.shape()[2];
            let n = b_data.shape()[2];

            assert_eq!(b_data.shape(), &[batch, k, n]);

            let mut out = ArrayD::zeros(vec![batch, m, n]);

            for b_idx in 0..batch {
                let a_mat = a_data
                    .slice(s![b_idx, .., ..])
                    .into_dimensionality::<Ix2>()
                    .unwrap();
                let b_mat = b_data
                    .slice(s![b_idx, .., ..])
                    .into_dimensionality::<Ix2>()
                    .unwrap();
                let mut out_mat = out
                    .slice_mut(s![b_idx, .., ..])
                    .into_dimensionality::<Ix2>()
                    .unwrap();
                out_mat.assign(&a_mat.dot(&b_mat));
            }

            out
        }

        // ---------- 2-D ⊗ 2-D ----------
        (2, 2) => {
            let a_mat = a_data.view().into_dimensionality::<Ix2>().unwrap();
            let b_mat = b_data.view().into_dimensionality::<Ix2>().unwrap();
            a_mat.dot(&b_mat).into_dyn()
        }

        _ => panic!(
            "dot_op: unsupported tensor dimensions: a={:?}, b={:?}",
            a_data.shape(),
            b_data.shape()
        ),
    };

    let requires_grad = a.grad.is_some() || b.grad.is_some();
    let mut result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let b_clone = b.clone();

        let backward_fn = Box::new(move |upstream_grad: &ArrayD<f32>| {
            let a_ref = a_clone.data.borrow();
            let b_ref = b_clone.data.borrow();

            match (a_ref.ndim(), b_ref.ndim()) {
                // ---------- 4-D ⊗ 4-D backward ----------
                (4, 4) => {
                    let batch = a_ref.shape()[0];
                    let heads = a_ref.shape()[1];
                    let m = a_ref.shape()[2];
                    let k = a_ref.shape()[3];
                    let n = b_ref.shape()[3];

                    let up = upstream_grad
                        .clone()
                        .into_dimensionality::<Ix4>()
                        .unwrap();

                    // dA = upstream · Bᵀ
                    if let Some(grad_a) = &a_clone.grad {
                        let mut ga = grad_a.borrow_mut();
                        for b in 0..batch {
                            for h in 0..heads {
                                let up_mat = up.slice(s![b, h, .., ..]);
                                let b_mat = b_ref.slice(s![b, h, .., ..]);
                                ga.slice_mut(s![b, h, .., ..])
                                    .add_assign(&up_mat.dot(&b_mat.t()));
                            }
                        }
                    }

                    // dB = Aᵀ · upstream
                    if let Some(grad_b) = &b_clone.grad {
                        let mut gb = grad_b.borrow_mut();
                        for b in 0..batch {
                            for h in 0..heads {
                                let a_mat = a_ref.slice(s![b, h, .., ..]);
                                let up_mat = up.slice(s![b, h, .., ..]);
                                gb.slice_mut(s![b, h, .., ..])
                                    .add_assign(&a_mat.t().dot(&up_mat));
                            }
                        }
                    }
                }

                // ---------- 3-D ⊗ 2-D backward ----------
                (3, 2) => {
                    // (код из предыдущего ответа)
                }

                // ---------- 3-D ⊗ 3-D backward ----------
                (3, 3) => {
                    // (код из предыдущего ответа)
                }

                // ---------- 2-D ⊗ 2-D backward ----------
                (2, 2) => {
                    // (код из предыдущего ответа)
                }

                _ => unreachable!(),
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a.clone(), b.clone()],
            backward_fn,
        }));
    }

    result
}