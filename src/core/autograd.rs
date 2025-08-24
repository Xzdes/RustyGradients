use crate::tensor::Tensor;
use ndarray::ArrayD;
use std::collections::HashSet;
use std::fmt;
use std::rc::Rc;

/// Контекст, который создается в результате операции.
/// Он содержит "указатели" на входные тензоры и функцию,
/// которая знает, как рассчитать и распространить градиент
/// на эти входы.
pub struct BackwardContext {
    pub inputs: Vec<Tensor>,
    pub backward_fn: Box<dyn Fn(&ArrayD<f32>)>,
}

impl fmt::Debug for BackwardContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BackwardContext")
            .field("num_inputs", &self.inputs.len())
            .finish()
    }
}

/// Запускает обратное распространение ошибки, начиная с заданного тензора.
/// Эта функция вынесена из самого `Tensor`, чтобы отделить данные от логики графа.
pub fn backward(tensor: &Tensor) {
    // --- Шаг 1: Построение графа и топологическая сортировка ---
    let mut sorted_graph = Vec::new();
    let mut visited = HashSet::new();

    fn build_graph(tensor: &Tensor, visited: &mut HashSet<Tensor>, sorted: &mut Vec<Tensor>) {
        if visited.contains(tensor) {
            return;
        }
        visited.insert(tensor.clone());

        // Если у тензора есть контекст (т.е. он результат какой-то операции),
        // то сначала рекурсивно обходим его входы.
        if let Some(ctx) = &tensor.ctx {
            for input_tensor in &ctx.inputs {
                build_graph(input_tensor, visited, sorted);
            }
        }
        // После того как все "дети" узла были обработаны, добавляем сам узел в отсортированный список.
        sorted.push(tensor.clone());
    }

    build_graph(tensor, &mut visited, &mut sorted_graph);

    // --- Шаг 2: Обратное распространение по отсортированному графу ---

    // Инициализируем градиент для самого последнего тензора (на котором вызвали backward).
    // Обычно это скаляр (loss), и его градиент по отношению к самому себе равен 1.
    if let Some(grad) = &tensor.grad {
        grad.borrow_mut().fill(1.0);
    } else {
        // Нельзя вызывать backward на тензоре, которому не нужен градиент.
        panic!("backward() called on a tensor that does not require gradients");
    }

    // Итерируемся по графу в обратном порядке (от конца к началу).
    for t in sorted_graph.iter().rev() {
        // Если у тензора есть контекст...
        if let Some(ctx) = &t.ctx {
            // ...берем его градиент (который к этому моменту уже должен быть вычислен)...
            let upstream_grad = t.grad.as_ref().unwrap().borrow();
            // ...и вызываем функцию обратного прохода, которая распространит этот градиент на входы.
            (ctx.backward_fn)(&upstream_grad);
        }
    }
}