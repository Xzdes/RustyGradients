// src/ops/selection.rs
use crate::core::autograd::BackwardContext;
use crate::tensor::Tensor;
use ndarray::Axis;
use std::rc::Rc;

/// Выполняет Softmax и Cross-Entropy за один проход для стабильности и корректного градиента.
/// Эта операция объединяет в себе несколько шагов, чтобы обеспечить математическую корректность
/// обратного распространения ошибки.
///
/// # Аргументы
///
/// * `logits` - тензор сырых, ненормализованных предсказаний от модели.
///   Ожидаемая форма: `[batch_size, seq_len, vocab_size]`. В нашем случае batch_size = 1.
/// * `targets` - тензор с целочисленными индексами правильных классов (токенов).
///   Ожидаемая форма: `[seq_len]`.
///
/// # Возвращает
///
/// Скалярный тензор (0-мерный), содержащий среднее значение Cross-Entropy Loss.
pub fn sparse_cross_entropy_op(logits: &Tensor, targets: &Tensor) -> Tensor {
    let logits_data = logits.data.borrow();
    let targets_data = targets.data.borrow();
    let last_axis = Axis(logits_data.ndim() - 1);

    // --- Прямой проход (Log-Softmax + выборка нужных значений) ---

    // 1. Стабилизируем логиты, вычитая максимум (трюк Log-Sum-Exp)
    let max_logits = logits_data
        .map_axis(last_axis, |row| {
            row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        })
        .into_dyn()
        .insert_axis(last_axis);
    
    let stable_logits = &*logits_data - &max_logits;

    // 2. Вычисляем log-вероятности (log-softmax)
    let log_sum_exp = stable_logits
        .mapv(|v| v.exp())
        .sum_axis(last_axis)
        .mapv(|v| v.ln())
        .into_dyn()
        .insert_axis(last_axis);
    
    let log_probs = &stable_logits - &log_sum_exp;

    // 3. Выбираем log-вероятности для целевых токенов и суммируем их
    let seq_len = targets_data.len();
    let mut total_loss = 0.0;
    for i in 0..seq_len {
        let target_idx = targets_data[i] as usize;
        // Мы предполагаем batch_size = 1
        total_loss -= log_probs[[0, i, target_idx]];
    }
    // 4. Усредняем по всей последовательности
    let mean_loss = total_loss / seq_len as f32;
    
    let mut result = Tensor::new(ndarray::arr0(mean_loss).into_dyn(), logits.grad.is_some());

    // --- Обратный проход ---
    if logits.grad.is_some() {
        let logits_for_closure = logits.clone();
        let logits_for_inputs = logits.clone();
        let targets_for_closure = targets.clone();

        // Производная Cross-Entropy Loss по отношению к логитам это: softmax(logits) - one_hot(targets)
        let probabilities = log_probs.mapv(|v| v.exp());

        let backward_fn = Box::new(move |upstream_grad: &ndarray::ArrayD<f32>| {
            if let Some(logits_grad) = &logits_for_closure.grad {
                let upstream_scalar = *upstream_grad.first().expect("Upstream grad for loss must be a scalar");
                let seq_len_bw = targets_for_closure.data.borrow().len();

                // Начинаем с вероятностей softmax
                let mut d_logits = probabilities.clone();
                let targets_data_bw = targets_for_closure.data.borrow();
                
                // Вычитаем 1.0 в позициях правильных ответов (one-hot кодирование)
                for i in 0..seq_len_bw {
                    let target_idx = targets_data_bw[i] as usize;
                    d_logits[[0, i, target_idx]] -= 1.0;
                }
                
                // Масштабируем градиент на upstream_grad (обычно 1.0) и усредняем по последовательности
                d_logits.mapv_inplace(|v| v * upstream_scalar / seq_len_bw as f32);

                // Добавляем вычисленный градиент к существующему
                logits_grad.borrow_mut().scaled_add(1.0, &d_logits);
            }
        });

        result.ctx = Some(Rc::new(BackwardContext {
            inputs: vec![logits_for_inputs],
            backward_fn,
        }));
    }

    result
}