use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use std::rc::Rc;

// Реализация операции суммирования всех элементов.
pub fn sum_op(a: &Tensor) -> Tensor {
    // Результат - скаляр (0-мерный тензор)
    let result_data = ArrayD::from_elem(IxDyn(&[]), a.data.borrow().sum());
    let mut result = Tensor::new(result_data, a.grad.is_some());

    if a.grad.is_some() {
        let a_for_ctx = a.clone();
        let a_for_closure = a.clone();

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            // Градиент суммы - это upstream_grad (скаляр), "растянутый" до формы
            // исходного тензора `a`.
            if let Some(grad_a) = &a_for_closure.grad {
                let upstream_scalar = upstream_grad
                    .first()
                    .expect("Upstream grad for sum must be a scalar");
                
                // Мы прибавляем градиент, а не перезаписываем его.
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