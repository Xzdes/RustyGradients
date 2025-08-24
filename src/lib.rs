// Делаем `Tensor` доступным на верхнем уровне.
// `use slmrustai::tensor::Tensor`
pub mod tensor;

// Создаем публичные модули, чтобы пользователь мог писать, например, `slmrustai::nn::Linear`.
pub mod core;
pub mod nn;
pub mod ops;
pub mod optim;

// -- Закомментируем пустой модуль-заглушку для будущих функций потерь --
// pub mod losses;