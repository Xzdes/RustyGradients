//! TensorV2 - Device-agnostic tensor с multi-backend поддержкой
//!
//! Это новая версия Tensor, поддерживающая CPU/CUDA/Metal/WASM backends.
//! Постепенно заменит старый Tensor.

use crate::backend::{Backend, Device, DeviceType};
use crate::core::autograd::BackwardContext;
use crate::error::Result;
use ndarray::ArrayD;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::Arc;

/// Типы данных для тензоров
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    U32,
}

/// TensorData - хранилище данных, зависящее от устройства
pub enum TensorData {
    /// CPU storage (ndarray)
    Cpu(ArrayD<f32>),

    /// Candle storage (для CUDA/Metal через candle-core)
    #[cfg(feature = "candle")]
    Candle(candle_core::Tensor),

    /// Placeholder для других backend
    #[allow(dead_code)]
    Opaque(Vec<u8>),
}

impl TensorData {
    pub fn shape(&self) -> Vec<usize> {
        match self {
            TensorData::Cpu(arr) => arr.shape().to_vec(),
            #[cfg(feature = "candle")]
            TensorData::Candle(tensor) => tensor.dims().to_vec(),
            TensorData::Opaque(_) => vec![],
        }
    }

    /// Конвертировать в CPU ndarray для backward compatibility
    pub fn to_cpu_array(&self) -> Result<ArrayD<f32>> {
        match self {
            TensorData::Cpu(arr) => Ok(arr.clone()),
            #[cfg(feature = "candle")]
            TensorData::Candle(tensor) => {
                // TODO: Implement candle -> ndarray conversion
                unimplemented!("Candle to CPU conversion not yet implemented")
            }
            TensorData::Opaque(_) => unimplemented!("Opaque tensor conversion"),
        }
    }
}

/// TensorV2 - новая версия с device support
#[derive(Clone)]
pub struct TensorV2 {
    /// Внутренние данные (device-specific)
    data: Arc<TensorData>,

    /// Градиент (lazy allocation)
    grad: Option<Arc<TensorData>>,

    /// Контекст для backward pass
    pub ctx: Option<Rc<BackwardContext>>,

    /// Устройство, на котором находится тензор
    device: Device,

    /// Тип данных
    dtype: DType,

    /// Требуется ли градиент
    requires_grad: bool,
}

impl TensorV2 {
    /// Создать новый TensorV2 на заданном устройстве
    pub fn new(data: ArrayD<f32>, requires_grad: bool, device: Device) -> Self {
        let data = Arc::new(TensorData::Cpu(data));

        Self {
            data,
            grad: None,
            ctx: None,
            device,
            dtype: DType::F32,
            requires_grad,
        }
    }

    /// Создать тензор на CPU (для backward compatibility)
    pub fn new_cpu(data: ArrayD<f32>, requires_grad: bool) -> Self {
        Self::new(data, requires_grad, Device::cpu())
    }

    /// Создать zeros тензор
    pub fn zeros(shape: &[usize], requires_grad: bool, device: Device) -> Result<Self> {
        let data = match device.device_type() {
            DeviceType::Cpu => {
                let arr = ArrayD::zeros(ndarray::IxDyn(shape));
                TensorData::Cpu(arr)
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                // TODO: Use backend to create zeros
                unimplemented!("CUDA zeros not yet implemented")
            }
            #[cfg(feature = "metal-backend")]
            DeviceType::Metal => {
                unimplemented!("Metal zeros not yet implemented")
            }
            #[cfg(target_arch = "wasm32")]
            DeviceType::Wasm => {
                unimplemented!("WASM zeros not yet implemented")
            }
        };

        Ok(Self {
            data: Arc::new(data),
            grad: None,
            ctx: None,
            device,
            dtype: DType::F32,
            requires_grad,
        })
    }

    /// Создать ones тензор
    #[allow(unreachable_patterns)]
    pub fn ones(shape: &[usize], requires_grad: bool, device: Device) -> Result<Self> {
        let data = match device.device_type() {
            DeviceType::Cpu => {
                let arr = ArrayD::ones(ndarray::IxDyn(shape));
                TensorData::Cpu(arr)
            }
            _ => unimplemented!("Non-CPU ones not yet implemented"),
        };

        Ok(Self {
            data: Arc::new(data),
            grad: None,
            ctx: None,
            device,
            dtype: DType::F32,
            requires_grad,
        })
    }

    /// Получить форму тензора
    pub fn shape(&self) -> Vec<usize> {
        self.data.shape()
    }

    /// Получить устройство
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Получить dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Требуется ли градиент
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Перенести тензор на другое устройство
    pub fn to_device(&self, target_device: &Device) -> Result<Self> {
        if self.device.device_type() == target_device.device_type() {
            return Ok(self.clone());
        }

        // TODO: Implement device transfer via backend
        unimplemented!("Device transfer not yet implemented")
    }

    /// Получить CPU данные (для backward compatibility)
    pub fn to_cpu_data(&self) -> Result<ArrayD<f32>> {
        self.data.to_cpu_array()
    }

    /// Доступ к градиенту (если есть)
    pub fn grad(&self) -> Option<&Arc<TensorData>> {
        self.grad.as_ref()
    }

    /// Обнулить градиент
    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            self.grad = None;
        }
    }

    // === Operations (будут делегированы backend) ===

    /// Сложение
    pub fn add(&self, other: &TensorV2) -> Result<TensorV2> {
        // TODO: Delegate to backend
        // Пока используем CPU fallback
        if let (TensorData::Cpu(a), TensorData::Cpu(b)) = (&*self.data, &*other.data) {
            let result = a + b;
            Ok(TensorV2::new_cpu(result, self.requires_grad || other.requires_grad))
        } else {
            unimplemented!("Non-CPU add not yet implemented")
        }
    }

    /// Вычитание
    pub fn sub(&self, other: &TensorV2) -> Result<TensorV2> {
        if let (TensorData::Cpu(a), TensorData::Cpu(b)) = (&*self.data, &*other.data) {
            let result = a - b;
            Ok(TensorV2::new_cpu(result, self.requires_grad || other.requires_grad))
        } else {
            unimplemented!("Non-CPU sub not yet implemented")
        }
    }

    /// Умножение
    pub fn mul(&self, other: &TensorV2) -> Result<TensorV2> {
        if let (TensorData::Cpu(a), TensorData::Cpu(b)) = (&*self.data, &*other.data) {
            let result = a * b;
            Ok(TensorV2::new_cpu(result, self.requires_grad || other.requires_grad))
        } else {
            unimplemented!("Non-CPU mul not yet implemented")
        }
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &TensorV2) -> Result<TensorV2> {
        // Используем backend для вычислений
        use crate::backend::cpu::CpuBackend;

        if let (TensorData::Cpu(a), TensorData::Cpu(b)) = (&*self.data, &*other.data) {
            let backend = CpuBackend::new();
            let result = backend.matmul(a, b)?;
            Ok(TensorV2::new_cpu(result, self.requires_grad || other.requires_grad))
        } else {
            unimplemented!("Non-CPU matmul not yet implemented")
        }
    }

    /// ReLU activation
    pub fn relu(&self) -> Result<TensorV2> {
        use crate::backend::cpu::CpuBackend;

        if let TensorData::Cpu(a) = &*self.data {
            let backend = CpuBackend::new();
            let result = backend.relu(a)?;
            Ok(TensorV2::new_cpu(result, self.requires_grad))
        } else {
            unimplemented!("Non-CPU relu not yet implemented")
        }
    }

    /// Softmax
    pub fn softmax(&self) -> Result<TensorV2> {
        use crate::backend::cpu::CpuBackend;

        if let TensorData::Cpu(a) = &*self.data {
            let backend = CpuBackend::new();
            let result = backend.softmax(a)?;
            Ok(TensorV2::new_cpu(result, self.requires_grad))
        } else {
            unimplemented!("Non-CPU softmax not yet implemented")
        }
    }

    /// Reshape
    pub fn reshape(&self, new_shape: &[usize]) -> Result<TensorV2> {
        use crate::backend::cpu::CpuBackend;

        if let TensorData::Cpu(a) = &*self.data {
            let backend = CpuBackend::new();
            let result = backend.reshape(a, new_shape)?;
            Ok(TensorV2::new_cpu(result, self.requires_grad))
        } else {
            unimplemented!("Non-CPU reshape not yet implemented")
        }
    }

    /// Transpose
    pub fn transpose(&self, axis1: usize, axis2: usize) -> Result<TensorV2> {
        use crate::backend::cpu::CpuBackend;

        if let TensorData::Cpu(a) = &*self.data {
            let backend = CpuBackend::new();
            let result = backend.transpose(a, axis1, axis2)?;
            Ok(TensorV2::new_cpu(result, self.requires_grad))
        } else {
            unimplemented!("Non-CPU transpose not yet implemented")
        }
    }
}

// Hash и PartialEq для использования в HashSet (autograd)
impl Hash for TensorV2 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.data).hash(state);
    }
}

impl PartialEq for TensorV2 {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }
}

impl Eq for TensorV2 {}

impl fmt::Debug for TensorV2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TensorV2 {{ shape: {:?}, device: {:?}, dtype: {:?}, requires_grad: {} }}",
            self.shape(),
            self.device.device_type(),
            self.dtype,
            self.requires_grad
        )
    }
}

// Convenience constructors
impl TensorV2 {
    /// Random normal distribution (CPU only for now)
    pub fn randn(shape: &[usize], requires_grad: bool) -> Self {
        use ndarray_rand::rand_distr::StandardNormal;
        use ndarray_rand::RandomExt;

        let arr = ArrayD::random(ndarray::IxDyn(shape), StandardNormal);
        Self::new_cpu(arr, requires_grad)
    }

    /// From slice
    pub fn from_slice(data: &[f32], shape: &[usize], requires_grad: bool) -> Result<Self> {
        let arr = ArrayD::from_shape_vec(ndarray::IxDyn(shape), data.to_vec())
            .map_err(|e| crate::error::RustyGradientsError::ShapeError(e.to_string()))?;
        Ok(Self::new_cpu(arr, requires_grad))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let device = Device::cpu();
        let t = TensorV2::zeros(&[2, 3], true, device).unwrap();
        assert_eq!(t.shape(), vec![2, 3]);
        assert!(t.requires_grad());
    }

    #[test]
    fn test_tensor_operations() {
        let a = TensorV2::ones(&[2, 2], false, Device::cpu()).unwrap();
        let b = TensorV2::ones(&[2, 2], false, Device::cpu()).unwrap();

        let c = a.add(&b).unwrap();
        assert_eq!(c.shape(), vec![2, 2]);
    }

    #[test]
    fn test_matmul() {
        let a = TensorV2::randn(&[3, 4], false);
        let b = TensorV2::randn(&[4, 5], false);

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), vec![3, 5]);
    }
}
