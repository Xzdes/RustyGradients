use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
use ndarray::Ix2;
use std::rc::Rc;

// Реализация матричного умножения.
pub fn dot_op(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data_borrow = a.data.borrow();
    let b_data_borrow = b.data.borrow();

    let a_view = a_data_borrow
        .view()
        .into_dimensionality::<Ix2>()
        .expect("Input 'a' to dot product must be 2D");
    let b_view = b_data_borrow
        .view()
        .into_dimensionality::<Ix2>()
        .expect("Input 'b' to dot product must be 2D");

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
            let a_view_closure = a_data_borrow_closure
                .view()
                .into_dimensionality::<Ix2>()
                .unwrap();
            let b_view_closure = b_data_borrow_closure
                .view()
                .into_dimensionality::<Ix2>()
                .unwrap();
            let upstream_view = upstream_grad.view().into_dimensionality::<Ix2>().unwrap();

            // grad_a = upstream_grad.dot(b^T)
            if let Some(grad_a) = &a_for_closure.grad {
                let grad_a_update = upstream_view.dot(&b_view_closure.t());
                grad_a
                    .borrow_mut()
                    .scaled_add(1.0, &grad_a_update.into_dyn());
            }
            // grad_b = a^T.dot(upstream_grad)
            if let Some(grad_b) = &b_for_closure.grad {
                let grad_b_update = a_view_closure.t().dot(&upstream_view);
                grad_b
                    .borrow_mut()
                    .scaled_add(1.0, &grad_b_update.into_dyn());
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![a_for_ctx, b_for_ctx],
            backward_fn,
        }));
    }
    result
}