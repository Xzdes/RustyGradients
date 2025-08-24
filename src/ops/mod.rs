// Объявляем файлы как публичные субмодули внутри `ops`.
// Теперь мы сможем обращаться к ним, например, как `crate::ops::basic_ops`.
pub mod basic_ops;
pub mod elementwise;
pub mod matmul;
pub mod reduction;