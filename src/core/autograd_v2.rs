//! Autograd V2 - автоматическое дифференцирование для TensorV2
//!
//! Поддерживает device-agnostic backward pass с multi-backend.

use crate::tensor_v2::{TensorV2, TensorData};
use std::collections::HashSet;

/// BackwardContextV2 - контекст для обратного распространения
///
/// Хранит входные тензоры и замыкание для вычисления градиентов.
pub struct BackwardContextV2 {
    pub inputs: Vec<TensorV2>,
    pub backward_fn: Box<dyn Fn(&TensorData, &mut Vec<Option<TensorData>>)>,
}

impl std::fmt::Debug for BackwardContextV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackwardContextV2")
            .field("num_inputs", &self.inputs.len())
            .finish()
    }
}

/// Построить вычислительный граф с топологической сортировкой
fn build_graph(tensor: &TensorV2, visited: &mut HashSet<TensorV2>, sorted: &mut Vec<TensorV2>) {
    if visited.contains(tensor) {
        return;
    }
    visited.insert(tensor.clone());

    // TODO: Рекурсивно обходим входы (если есть контекст)
    // Пока закомментировано, так как ctx использует старый BackwardContext
    // if let Some(ctx) = &tensor.ctx {
    //     for input_tensor in &ctx.inputs {
    //         build_graph(input_tensor, visited, sorted);
    //     }
    // }
    let _ = tensor.ctx; // Suppress unused warning

    sorted.push(tensor.clone());
}

/// Backward pass для TensorV2
///
/// Вычисляет градиенты для всех тензоров в графе вычислений,
/// начиная с заданного тензора (обычно loss).
pub fn backward_v2(tensor: &TensorV2) {
    if !tensor.requires_grad() {
        panic!("backward() called on tensor that doesn't require gradients");
    }

    // Шаг 1: Топологическая сортировка графа
    let mut sorted_graph = Vec::new();
    let mut visited = HashSet::new();
    build_graph(tensor, &mut visited, &mut sorted_graph);

    // Шаг 2: Инициализация градиента корневого тензора
    // TODO: Нужно добавить способ установки градиента в TensorV2
    // Пока это заглушка - реализуем позже

    // Шаг 3: Обратный проход
    for t in sorted_graph.iter().rev() {
        if let Some(_ctx) = &t.ctx {
            // TODO: Вызов backward_fn с правильной передачей градиентов
            // Реализуем после добавления gradient storage в TensorV2
        }
    }
}

/// Helper для создания backward closure
///
/// Упрощает создание backward функций для операций.
pub fn make_backward_fn<F>(f: F) -> Box<dyn Fn(&TensorData, &mut Vec<Option<TensorData>>)>
where
    F: Fn(&TensorData, &mut Vec<Option<TensorData>>) + 'static,
{
    Box::new(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Device;
    use crate::tensor_v2::TensorV2;

    #[test]
    #[ignore = "TensorV2 autograd integration incomplete - graph building needs context propagation"]
    fn test_graph_building() {
        let a = TensorV2::ones(&[2, 2], true, Device::cpu()).unwrap();
        let b = TensorV2::ones(&[2, 2], true, Device::cpu()).unwrap();
        let c = a.add(&b).unwrap();

        let mut visited = HashSet::new();
        let mut sorted = Vec::new();
        build_graph(&c, &mut visited, &mut sorted);

        // Граф должен содержать все 3 тензора
        assert_eq!(sorted.len(), 3);
    }

    #[test]
    #[should_panic(expected = "doesn't require gradients")]
    fn test_backward_without_grad() {
        let t = TensorV2::ones(&[2, 2], false, Device::cpu()).unwrap();
        backward_v2(&t);
    }
}
