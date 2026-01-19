//! Operations V2 - операции с autograd для TensorV2
//!
//! Все операции автоматически поддерживают backward pass.

pub mod basic;
pub mod matmul;

// Re-export основных операций
pub use basic::{add_op, mul_op, sub_op};
pub use matmul::matmul_op;
