use crate::tensor::Tensor;
use crate::core::autograd::BackwardContext;
use ndarray::{Axis, Order};
use std::ops::{Add, Mul, Sub};
use std::rc::Rc;

// Реализуем трейт `Add` с корректной обработкой "вещания" (broadcasting) в обратном проходе.
impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Self::Output {
        let result_data = &*self.data.borrow() + &*rhs.data.borrow();
        let requires_grad = self.grad.is_some() || rhs.grad.is_some();
        let mut result = Tensor::new(result_data, requires_grad);

        if requires_grad {
            let a_for_ctx = self.clone();
            let b_for_ctx = rhs.clone();
            let a_for_closure = self.clone();
            let b_for_closure = rhs.clone();

            let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
                // Градиент для 'a'
                if let Some(grad_a) = &a_for_closure.grad {
                    let mut grad_a_borrow = grad_a.borrow_mut();
                    if grad_a_borrow.shape() != upstream_grad.shape() {
                        let summed_grad = upstream_grad.sum_axis(Axis(0));
                        let reshaped_grad =
                            summed_grad.into_shape_with_order(grad_a_borrow.shape()).unwrap();
                        *grad_a_borrow += &reshaped_grad.into_dyn();
                    } else {
                        *grad_a_borrow += upstream_grad;
                    }
                }
                // Градиент для 'b'
                if let Some(grad_b) = &b_for_closure.grad {
                    let mut grad_b_borrow = grad_b.borrow_mut();
                    if grad_b_borrow.shape() != upstream_grad.shape() {
                        let summed_grad = upstream_grad.sum_axis(Axis(0));
                        let reshaped_grad =
                            summed_grad.into_shape_with_order(grad_b_borrow.shape()).unwrap();
                        *grad_b_borrow += &reshaped_grad.into_dyn();
                    } else {
                        *grad_b_borrow += upstream_grad;
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
}

// Реализуем трейт `Mul` для поэлементного умножения.
impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        let result_data = &*self.data.borrow() * &*rhs.data.borrow();
        let requires_grad = self.grad.is_some() || rhs.grad.is_some();
        let mut result = Tensor::new(result_data, requires_grad);

        if requires_grad {
            let a_for_ctx = self.clone();
            let b_for_ctx = rhs.clone();
            let a_for_closure = self.clone();
            let b_for_closure = rhs.clone();

            let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
                if let Some(grad_a) = &a_for_closure.grad {
                    let b_data = b_for_closure.data.borrow();
                    *grad_a.borrow_mut() += &(upstream_grad * &*b_data);
                }
                if let Some(grad_b) = &b_for_closure.grad {
                    let a_data = a_for_closure.data.borrow();
                    *grad_b.borrow_mut() += &(upstream_grad * &*a_data);
                }
            });

            result.ctx = Some(Rc::new(BackwardContext {
                inputs: vec![a_for_ctx, b_for_ctx],
                backward_fn,
            }));
        }
        result
    }
}

// Реализация операции вычитания.
impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        let result_data = &*self.data.borrow() - &*rhs.data.borrow();
        let requires_grad = self.grad.is_some() || rhs.grad.is_some();
        let mut result = Tensor::new(result_data, requires_grad);

        if requires_grad {
            let a_for_ctx = self.clone();
            let b_for_ctx = rhs.clone();
            let a_for_closure = self.clone();
            let b_for_closure = rhs.clone();

            let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
                if let Some(grad_a) = &a_for_closure.grad {
                    *grad_a.borrow_mut() += upstream_grad;
                }
                if let Some(grad_b) = &b_for_closure.grad {
                    *grad_b.borrow_mut() -= upstream_grad;
                }
            });

            result.ctx = Some(Rc::new(BackwardContext {
                inputs: vec![a_for_ctx, b_for_ctx],
                backward_fn,
            }));
        }
        result
    }
}