///! Backend abstraction layer для multi-device поддержки
///!
///! Этот модуль определяет интерфейс Backend для поддержки различных
///! вычислительных устройств (CPU, CUDA, Metal, WASM).

use crate::error::Result;
use ndarray::ArrayD;
use std::sync::Arc;

pub mod cpu;
pub mod simd; // SIMD-optimized elementwise operations
pub mod fused; // Fused multi-operation kernels

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

/// Тип устройства для выполнения вычислений
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU (всегда доступен)
    Cpu,
    /// CUDA GPU (требует feature = "cuda")
    #[cfg(feature = "cuda")]
    Cuda(usize), // GPU index
    /// Metal (Apple Silicon, требует feature = "metal")
    #[cfg(feature = "metal")]
    Metal,
    /// WebAssembly
    #[cfg(target_arch = "wasm32")]
    Wasm,
}

/// Device - обертка над backend для удобного использования
#[derive(Clone)]
pub struct Device {
    device_type: DeviceType,
    // Используем enum dispatch вместо dyn trait для лучшей производительности
    #[allow(dead_code)]
    backend: BackendImpl,
}

/// Enum dispatch для backend implementations
#[derive(Clone)]
#[allow(dead_code)]
enum BackendImpl {
    Cpu(Arc<cpu::CpuBackend>),
    #[cfg(feature = "cuda")]
    Cuda(Arc<cuda::CudaBackend>),
    #[cfg(feature = "metal-backend")]
    Metal(Arc<metal::MetalBackend>),
    #[cfg(target_arch = "wasm32")]
    Wasm(Arc<wasm::WasmBackend>),
}

impl Device {
    /// Создать CPU device
    pub fn cpu() -> Self {
        Self {
            device_type: DeviceType::Cpu,
            backend: BackendImpl::Cpu(Arc::new(cpu::CpuBackend::new())),
        }
    }

    /// Создать CUDA device
    #[cfg(feature = "cuda")]
    pub fn cuda(index: usize) -> Result<Self> {
        Ok(Self {
            device_type: DeviceType::Cuda(index),
            backend: BackendImpl::Cuda(Arc::new(cuda::CudaBackend::new(index)?)),
        })
    }

    /// Создать Metal device
    #[cfg(feature = "metal-backend")]
    pub fn metal() -> Result<Self> {
        Ok(Self {
            device_type: DeviceType::Metal,
            backend: BackendImpl::Metal(Arc::new(metal::MetalBackend::new()?)),
        })
    }

    /// Создать WASM device
    #[cfg(target_arch = "wasm32")]
    pub fn wasm() -> Self {
        Self {
            device_type: DeviceType::Wasm,
            backend: BackendImpl::Wasm(Arc::new(wasm::WasmBackend::new())),
        }
    }

    /// Получить default device (автовыбор лучшего доступного)
    pub fn default_device() -> Self {
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Self::cuda(0) {
                return device;
            }
        }

        #[cfg(all(feature = "metal-backend", target_os = "macos"))]
        {
            if let Ok(device) = Self::metal() {
                return device;
            }
        }

        Self::cpu()
    }

    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    #[allow(dead_code, private_interfaces)]
    pub(crate) fn backend(&self) -> &BackendImpl {
        &self.backend
    }
}

/// Backend trait - основной интерфейс для всех бэкендов
///
/// Каждый бэкенд (CPU, CUDA, Metal, WASM) должен реализовать этот trait
/// для поддержки всех необходимых операций.
pub trait Backend: Send + Sync {
    /// Тип хранилища данных (зависит от бэкенда)
    type Storage: Send + Sync;

    /// Получить тип устройства
    fn device_type(&self) -> DeviceType;

    /// Синхронизировать выполнение (важно для GPU)
    fn synchronize(&self) -> Result<()>;

    // === Memory Operations ===

    /// Выделить память заданной формы, заполненную нулями
    fn zeros(&self, shape: &[usize]) -> Result<Self::Storage>;

    /// Выделить память заданной формы, заполненную единицами
    fn ones(&self, shape: &[usize]) -> Result<Self::Storage>;

    /// Создать storage из слайса данных
    fn from_slice(&self, data: &[f32], shape: &[usize]) -> Result<Self::Storage>;

    /// Скопировать данные обратно в Vec<f32> (на host)
    fn to_vec(&self, storage: &Self::Storage) -> Result<Vec<f32>>;

    /// Получить форму storage
    fn shape(&self, storage: &Self::Storage) -> Vec<usize>;

    // === Arithmetic Operations ===

    /// Поэлементное сложение с broadcasting
    fn add(&self, a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage>;

    /// Поэлементное вычитание с broadcasting
    fn sub(&self, a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage>;

    /// Поэлементное умножение с broadcasting
    fn mul(&self, a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage>;

    /// Матричное умножение (batched dot product)
    fn matmul(&self, a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage>;

    // === Element-wise Operations ===

    /// ReLU активация: max(0, x)
    fn relu(&self, a: &Self::Storage) -> Result<Self::Storage>;

    /// Sigmoid активация: 1 / (1 + exp(-x))
    fn sigmoid(&self, a: &Self::Storage) -> Result<Self::Storage>;

    /// Экспонента: exp(x)
    fn exp(&self, a: &Self::Storage) -> Result<Self::Storage>;

    /// Натуральный логарифм: ln(x)
    fn log(&self, a: &Self::Storage) -> Result<Self::Storage>;

    /// Возведение в степень: x^power
    fn powf(&self, a: &Self::Storage, power: f32) -> Result<Self::Storage>;

    /// Softmax по последней оси
    fn softmax(&self, a: &Self::Storage) -> Result<Self::Storage>;

    // === Reduction Operations ===

    /// Суммирование всех элементов
    fn sum(&self, a: &Self::Storage) -> Result<Self::Storage>;

    /// Суммирование по заданной оси
    fn sum_axis(&self, a: &Self::Storage, axis: usize) -> Result<Self::Storage>;

    // === Transformation Operations ===

    /// Изменить форму тензора
    fn reshape(&self, a: &Self::Storage, new_shape: &[usize]) -> Result<Self::Storage>;

    /// Транспонировать две оси
    fn transpose(&self, a: &Self::Storage, axis1: usize, axis2: usize) -> Result<Self::Storage>;

    // === Special Operations ===

    /// Embedding lookup: weights[indices]
    fn embedding(&self, indices: &Self::Storage, weights: &Self::Storage) -> Result<Self::Storage>;

    /// Layer Normalization
    fn layer_norm(
        &self,
        x: &Self::Storage,
        gamma: &Self::Storage,
        beta: &Self::Storage,
        epsilon: f32,
    ) -> Result<Self::Storage>;

    /// Sparse Cross-Entropy Loss
    fn sparse_cross_entropy(
        &self,
        logits: &Self::Storage,
        targets: &Self::Storage,
    ) -> Result<Self::Storage>;
}

/// Storage wrapper - универсальная обертка над backend-specific storage
pub enum Storage {
    Cpu(ArrayD<f32>),
    #[cfg(feature = "cuda")]
    Cuda(cuda::CudaStorage),
    #[cfg(feature = "metal")]
    Metal(metal::MetalStorage),
    #[cfg(target_arch = "wasm32")]
    Wasm(wasm::WasmStorage),
}

impl Storage {
    pub fn shape(&self) -> &[usize] {
        match self {
            Storage::Cpu(arr) => arr.shape(),
            #[cfg(feature = "cuda")]
            Storage::Cuda(storage) => &storage.shape,
            #[cfg(feature = "metal")]
            Storage::Metal(storage) => &storage.shape,
            #[cfg(target_arch = "wasm32")]
            Storage::Wasm(storage) => &storage.shape,
        }
    }
}
