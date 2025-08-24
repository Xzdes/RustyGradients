use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
use std::rc::Rc;

// Реализация операции возведения в степень.
pub fn powf_op(a: &Tensor, power: f32) -> Tensor {
    let result_data = a.data.borrow().mapv(|val| val.powf(power));
    let mut result = Tensor::new(result_data, a.grad.is_some());

    if a.grad.is_some() {
        let a_for_ctx = a.clone();
        let a_for_closure = a.clone();

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            // grad_a = upstream_grad * power * a^(power - 1)
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