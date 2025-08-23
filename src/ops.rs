use crate::tensor::{BackwardContext, Tensor};
// Убираем неиспользуемый `Order` из импортов.
use ndarray::{ArrayD, Axis, Ix2, IxDyn};
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
                        // --- ИСПРАВЛЕНИЕ: Убираем второй аргумент ---
                        let reshaped_grad = summed_grad.into_shape_with_order(grad_a_borrow.shape()).unwrap();
                        *grad_a_borrow += &reshaped_grad.into_dyn();
                    } else {
                        *grad_a_borrow += upstream_grad;
                    }
                }
                // Градиент для 'b' (аналогично для bias)
                if let Some(grad_b) = &b_for_closure.grad {
                    let mut grad_b_borrow = grad_b.borrow_mut();
                    if grad_b_borrow.shape() != upstream_grad.shape() {
                        let summed_grad = upstream_grad.sum_axis(Axis(0));
                        // --- ИСПРАВЛЕНИЕ: Убираем второй аргумент ---
                        let reshaped_grad = summed_grad.into_shape_with_order(grad_b_borrow.shape()).unwrap();
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

pub fn powf_op(a: &Tensor, power: f32) -> Tensor {
    let result_data = a.data.borrow().mapv(|val| val.powf(power));
    let mut result = Tensor::new(result_data, a.grad.is_some());

    if a.grad.is_some() {
        let a_for_ctx = a.clone();
        let a_for_closure = a.clone();

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            let a_data = a_for_closure.data.borrow();
            let derivative = a_data.mapv(|val| power * val.powf(power - 1.0));
            if let Some(grad_a) = &a_for_closure.grad {
                *grad_a.borrow_mut() += &(upstream_grad * &derivative);
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_ctx],
            backward_fn,
        }));
    }
    result
}

pub fn sum_op(a: &Tensor) -> Tensor {
    let result_data = ArrayD::from_elem(IxDyn(&[]), a.data.borrow().sum());
    let mut result = Tensor::new(result_data, a.grad.is_some());

    if a.grad.is_some() {
        let a_for_ctx = a.clone();
        let a_for_closure = a.clone();

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            if let Some(grad_a) = &a_for_closure.grad {
                let upstream_scalar = upstream_grad.first().expect("Upstream grad for sum must be a scalar");
                let mut grad_a_borrow = grad_a.borrow_mut();
                for elem in grad_a_borrow.iter_mut() {
                    *elem += upstream_scalar;
                }
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_ctx],
            backward_fn,
        }));
    }
    result
}

pub fn dot_op(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data_borrow = a.data.borrow();
    let b_data_borrow = b.data.borrow();
    let a_view = a_data_borrow.view().into_dimensionality::<Ix2>().expect("Input 'a' to dot product must be 2D");
    let b_view = b_data_borrow.view().into_dimensionality::<Ix2>().expect("Input 'b' to dot product must be 2D");
    let result_data = a_view.dot(&b_view).into_dyn();
    let requires_grad = a.grad.is_some() || b.grad.is_some();
    let mut result = Tensor::new(result_data, requires_grad);

    if requires_grad {
        let a_for_ctx = a.clone();
        let b_for_ctx = b.clone();
        let a_for_closure = a.clone();
        let b_for_closure = b.clone();

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            let a_data_borrow_closure = a_for_closure.data.borrow();
            let b_data_borrow_closure = b_for_closure.data.borrow();
            let a_view_closure = a_data_borrow_closure.view().into_dimensionality::<Ix2>().unwrap();
            let b_view_closure = b_data_borrow_closure.view().into_dimensionality::<Ix2>().unwrap();
            let upstream_view = upstream_grad.view().into_dimensionality::<Ix2>().unwrap();

            if let Some(grad_a) = &a_for_closure.grad {
                let grad_a_update = upstream_view.dot(&b_view_closure.t());
                grad_a.borrow_mut().scaled_add(1.0, &grad_a_update.into_dyn());
            }
            if let Some(grad_b) = &b_for_closure.grad {
                let grad_b_update = a_view_closure.t().dot(&upstream_view);
                grad_b.borrow_mut().scaled_add(1.0, &grad_b_update.into_dyn());
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_ctx, b_for_ctx],
            backward_fn,
        }));
    }
    result
}