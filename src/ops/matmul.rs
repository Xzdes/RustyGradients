//! Модуль, реализующий операцию матричного умножения (dot product).

use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
use crate::error::{Result, RustyGradientsError};
// --- ИЗМЕНЕНИЕ: Добавляем Ix3 и Ix4 для явного приведения типов ---
use ndarray::{s, ArrayD, ArrayView2, Ix2, Ix3, Ix4};
use std::ops::AddAssign;
use std::rc::Rc;

/// Выполняет матричное умножение (batched dot) для тензоров разной размерности.
///
/// Поддерживаемые комбинации:
/// - 4D ⊗ 4D → 4D
/// - 3D ⊗ 2D → 3D
/// - 3D ⊗ 3D → 3D
/// - 2D ⊗ 2D → 2D
///
/// Может вернуть ошибку `RustyGradientsError::ShapeError`, если внутренние
/// размерности матриц не совпадают.
pub fn dot_op(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_data = a.data.borrow();
    let b_data = b.data.borrow();

    let result_data = match (a_data.ndim(), b_data.ndim()) {
        (4, 4) => {
            let batch_size = a_data.shape()[0];
            let heads = a_data.shape()[1];
            let m = a_data.shape()[2];
            let k = a_data.shape()[3];
            let n = b_data.shape()[3];

            if b_data.shape() != &[batch_size, heads, k, n] {
                return Err(RustyGradientsError::ShapeError(format!(
                    "Incompatible shapes for 4D dot product: {:?} and {:?}",
                    a_data.shape(), b_data.shape()
                )));
            }

            let mut out = ArrayD::zeros(vec![batch_size, heads, m, n]);
            for b_idx in 0..batch_size {
                for h_idx in 0..heads {
                    let a_mat = a_data.slice(s![b_idx, h_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                    let b_mat = b_data.slice(s![b_idx, h_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                    let mut c_mat = out.slice_mut(s![b_idx, h_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                    c_mat.assign(&a_mat.dot(&b_mat));
                }
            }
            Ok(out)
        }
        (3, 2) => {
            let batch = a_data.shape()[0];
            let seq = a_data.shape()[1];
            let in_dim = a_data.shape()[2];
            let out_dim = b_data.shape()[1];

            if b_data.shape() != &[in_dim, out_dim] {
                return Err(RustyGradientsError::ShapeError(format!(
                    "Incompatible shapes for 3D-2D dot product: {:?} and {:?}",
                    a_data.shape(), b_data.shape()
                )));
            }

            let mut out = ArrayD::zeros(vec![batch, seq, out_dim]);
            for b_idx in 0..batch {
                let a_mat = a_data.slice(s![b_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                let b_mat = b_data.view().into_dimensionality::<Ix2>().unwrap();
                let mut out_mat = out.slice_mut(s![b_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                out_mat.assign(&a_mat.dot(&b_mat));
            }
            Ok(out)
        }
        (3, 3) => {
            let batch = a_data.shape()[0];
            let m = a_data.shape()[1];
            let k = a_data.shape()[2];
            let n = b_data.shape()[2];

            if b_data.shape() != &[batch, k, n] {
                return Err(RustyGradientsError::ShapeError(format!(
                    "Incompatible shapes for 3D-3D dot product: {:?} and {:?}",
                    a_data.shape(), b_data.shape()
                )));
            }

            let mut out = ArrayD::zeros(vec![batch, m, n]);
            for b_idx in 0..batch {
                let a_mat = a_data.slice(s![b_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                let b_mat = b_data.slice(s![b_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                let mut out_mat = out.slice_mut(s![b_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                out_mat.assign(&a_mat.dot(&b_mat));
            }
            Ok(out)
        }
        (2, 2) => {
            let a_mat = a_data.view().into_dimensionality::<Ix2>().unwrap();
            let b_mat = b_data.view().into_dimensionality::<Ix2>().unwrap();
            if a_mat.shape()[1] != b_mat.shape()[0] {
                return Err(RustyGradientsError::ShapeError(format!(
                    "Incompatible shapes for 2D dot product: {:?} and {:?}",
                    a_data.shape(), b_data.shape()
                )));
            }
            Ok(a_mat.dot(&b_mat).into_dyn())
        }
        _ => Err(RustyGradientsError::InvalidInput(format!(
            "Unsupported tensor dimensions for dot_op: a={:?}, b={:?}",
            a_data.shape(), b_data.shape()
        ))),
    }?;

    let requires_grad = a.grad.is_some() || b.grad.is_some();
    let mut result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        let a_for_closure = a.clone();
        let b_for_closure = b.clone();

        let backward_fn = Box::new(move |upstream_grad: &ArrayD<f32>| {
            let a_ref = a_for_closure.data.borrow();
            let b_ref = b_for_closure.data.borrow();
            
            match (a_ref.ndim(), b_ref.ndim()) {
                (4, 4) => {
                    let batch = a_ref.shape()[0];
                    let heads = a_ref.shape()[1];
                    // --- ИСПРАВЛЕНИЕ: Явно указываем целевой тип ---
                    let up = upstream_grad.view().into_dimensionality::<Ix4>().unwrap();

                    if let Some(grad_a) = &a_for_closure.grad {
                        let mut ga = grad_a.borrow_mut();
                        for b_idx in 0..batch {
                            for h_idx in 0..heads {
                                let up_mat: ArrayView2<f32> = up.slice(s![b_idx, h_idx, .., ..]).into_dimensionality().unwrap();
                                let b_mat: ArrayView2<f32> = b_ref.slice(s![b_idx, h_idx, .., ..]).into_dimensionality().unwrap();
                                ga.slice_mut(s![b_idx, h_idx, .., ..])
                                    .add_assign(&up_mat.dot(&b_mat.t()));
                            }
                        }
                    }

                    if let Some(grad_b) = &b_for_closure.grad {
                        let mut gb = grad_b.borrow_mut();
                        for b_idx in 0..batch {
                            for h_idx in 0..heads {
                                let a_mat: ArrayView2<f32> = a_ref.slice(s![b_idx, h_idx, .., ..]).into_dimensionality().unwrap();
                                let up_mat: ArrayView2<f32> = up.slice(s![b_idx, h_idx, .., ..]).into_dimensionality().unwrap();
                                gb.slice_mut(s![b_idx, h_idx, .., ..])
                                    .add_assign(&a_mat.t().dot(&up_mat));
                            }
                        }
                    }
                }
                (3, 2) => {
                    let batch = a_ref.shape()[0];
                    let up = upstream_grad.view().into_dimensionality::<Ix3>().unwrap();
                    let b_mat: ArrayView2<f32> = b_ref.view().into_dimensionality().unwrap();
                    if let Some(grad_a) = &a_for_closure.grad {
                        let mut ga = grad_a.borrow_mut();
                        for b_idx in 0..batch {
                            let up_slice: ArrayView2<f32> = up.slice(s![b_idx, .., ..]).into_dimensionality().unwrap();
                            ga.slice_mut(s![b_idx, .., ..]).add_assign(&up_slice.dot(&b_mat.t()));
                        }
                    }
                    if let Some(grad_b) = &b_for_closure.grad {
                        let mut gb = grad_b.borrow_mut();
                        for b_idx in 0..batch {
                            let a_slice: ArrayView2<f32> = a_ref.slice(s![b_idx, .., ..]).into_dimensionality().unwrap();
                            let up_slice: ArrayView2<f32> = up.slice(s![b_idx, .., ..]).into_dimensionality().unwrap();
                            gb.add_assign(&a_slice.t().dot(&up_slice));
                        }
                    }
                }
                (3, 3) => {
                     let batch = a_ref.shape()[0];
                     let up = upstream_grad.view().into_dimensionality::<Ix3>().unwrap();
                     if let Some(grad_a) = &a_for_closure.grad {
                         let mut ga = grad_a.borrow_mut();
                          for b_idx in 0..batch {
                             let up_mat: ArrayView2<f32> = up.slice(s![b_idx, .., ..]).into_dimensionality().unwrap();
                             let b_mat: ArrayView2<f32> = b_ref.slice(s![b_idx, .., ..]).into_dimensionality().unwrap();
                             ga.slice_mut(s![b_idx, .., ..]).add_assign(&up_mat.dot(&b_mat.t()));
                         }
                     }
                     if let Some(grad_b) = &b_for_closure.grad {
                         let mut gb = grad_b.borrow_mut();
                          for b_idx in 0..batch {
                             let a_mat: ArrayView2<f32> = a_ref.slice(s![b_idx, .., ..]).into_dimensionality().unwrap();
                             let up_mat: ArrayView2<f32> = up.slice(s![b_idx, .., ..]).into_dimensionality().unwrap();
                             gb.slice_mut(s![b_idx, .., ..]).add_assign(&a_mat.t().dot(&up_mat));
                         }
                     }
                }
                (2, 2) => {
                    let up: ArrayView2<f32> = upstream_grad.view().into_dimensionality().unwrap();
                    let a_mat: ArrayView2<f32> = a_ref.view().into_dimensionality().unwrap();
                    let b_mat: ArrayView2<f32> = b_ref.view().into_dimensionality().unwrap();
                    if let Some(grad_a) = &a_for_closure.grad {
                         grad_a.borrow_mut().add_assign(&up.dot(&b_mat.t()).into_dyn());
                    }
                    if let Some(grad_b) = &b_for_closure.grad {
                         grad_b.borrow_mut().add_assign(&a_mat.t().dot(&up).into_dyn());
                    }
                }
                _ => unreachable!(),
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a.clone(), b.clone()],
            backward_fn,
        }));
    }

    Ok(result)
}