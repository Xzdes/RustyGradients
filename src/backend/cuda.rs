///! CUDA Backend Implementation
///!
///! This module provides GPU acceleration using NVIDIA CUDA.
///!
///! Features:
///! - cuBLAS for matrix multiplication (50-100x speedup)
///! - Custom CUDA kernels for elementwise operations (20-50x speedup)
///! - Device memory management
///! - Automatic CPU ↔ GPU data transfer
///!
///! Expected Performance:
///! - Matmul (1024x1024): 500 GFLOPS (CPU) → 5,000+ GFLOPS (GPU) = 10x
///! - Elementwise ops: 20-50x vs CPU
///! - Training throughput: 10 tok/s → 1,000+ tok/s = 100x

use crate::error::{Result, RustyGradientsError};
use crate::backend::{Backend, DeviceType};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut};
use cudarc::cublas::{CudaBlas, Gemm};
use std::sync::Arc;

#[cfg(feature = "cuda")]
mod cuda_kernels_wrapper;

#[cfg(feature = "cuda")]
use cuda_kernels_wrapper::CudaKernels;

/// CUDA Storage - хранит данные на GPU
pub struct CudaStorage {
    pub(crate) data: CudaSlice<f32>,
    pub(crate) shape: Vec<usize>,
    device: Arc<CudaDevice>,
}

impl CudaStorage {
    /// Create new CUDA storage from host data
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>, device: Arc<CudaDevice>) -> Result<Self> {
        // Verify shape matches data length
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(RustyGradientsError::ShapeMismatch {
                expected: shape.clone(),
                actual: vec![data.len()],
                context: "Creating CUDA storage".to_string(),
            });
        }

        // Transfer data to GPU
        let cuda_data = device.htod_sync_copy(&data)
            .map_err(|e| RustyGradientsError::BackendError(format!("CUDA memory copy failed: {:?}", e)))?;

        Ok(Self {
            data: cuda_data,
            shape,
            device,
        })
    }

    /// Copy data back to host (CPU)
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        self.device.dtoh_sync_copy(&self.data)
            .map_err(|e| RustyGradientsError::BackendError(format!("CUDA to host copy failed: {:?}", e)))
    }

    /// Get total number of elements
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get device reference
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

/// CUDA Backend Implementation
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    blas: Arc<CudaBlas>,
    #[cfg(feature = "cuda")]
    kernels: Option<Arc<CudaKernels>>,
    device_index: usize,
}

impl CudaBackend {
    /// Create new CUDA backend for specific GPU
    pub fn new(device_index: usize) -> Result<Self> {
        // Initialize CUDA device
        let device = CudaDevice::new(device_index)
            .map_err(|e| RustyGradientsError::BackendError(
                format!("Failed to initialize CUDA device {}: {:?}", device_index, e)
            ))?;

        // Initialize cuBLAS
        let blas = CudaBlas::new(device.clone())
            .map_err(|e| RustyGradientsError::BackendError(
                format!("Failed to initialize cuBLAS: {:?}", e)
            ))?;

        println!("✅ CUDA Backend initialized on GPU {}", device_index);
        println!("   Device: {}", device.name());
        println!("   Memory: {:.2} GB", device.total_memory() as f64 / 1e9);

        // Try to load custom kernels (optional)
        #[cfg(feature = "cuda")]
        let kernels = match CudaKernels::new(Arc::new(device.clone())) {
            Ok(k) => {
                println!("✅ Custom CUDA kernels loaded");
                Some(Arc::new(k))
            }
            Err(e) => {
                println!("⚠️  Custom CUDA kernels not available: {:?}", e);
                println!("   Falling back to cuBLAS-only mode");
                None
            }
        };

        Ok(Self {
            device: Arc::new(device),
            blas: Arc::new(blas),
            #[cfg(feature = "cuda")]
            kernels,
            device_index,
        })
    }

    /// Get number of available CUDA devices
    pub fn device_count() -> usize {
        CudaDevice::count().unwrap_or(0)
    }

    /// Check if CUDA is available
    pub fn is_available() -> bool {
        Self::device_count() > 0
    }
}

impl Backend for CudaBackend {
    type Storage = CudaStorage;

    fn device_type(&self) -> DeviceType {
        DeviceType::Cuda(self.device_index)
    }

    fn synchronize(&self) -> Result<()> {
        self.device.synchronize()
            .map_err(|e| RustyGradientsError::BackendError(
                format!("CUDA synchronization failed: {:?}", e)
            ))
    }

    // === Memory Operations ===

    fn zeros(&self, shape: &[usize]) -> Result<Self::Storage> {
        let len: usize = shape.iter().product();
        let data = vec![0.0f32; len];
        CudaStorage::from_vec(data, shape.to_vec(), self.device.clone())
    }

    fn ones(&self, shape: &[usize]) -> Result<Self::Storage> {
        let len: usize = shape.iter().product();
        let data = vec![1.0f32; len];
        CudaStorage::from_vec(data, shape.to_vec(), self.device.clone())
    }

    fn from_slice(&self, data: &[f32], shape: &[usize]) -> Result<Self::Storage> {
        CudaStorage::from_vec(data.to_vec(), shape.to_vec(), self.device.clone())
    }

    fn to_vec(&self, storage: &Self::Storage) -> Result<Vec<f32>> {
        storage.to_vec()
    }

    fn shape(&self, storage: &Self::Storage) -> Vec<usize> {
        storage.shape.clone()
    }

    // === Arithmetic Operations ===

    fn add(&self, a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref kernels) = self.kernels {
                // Use custom CUDA kernel (20-30x speedup expected)
                let n = a.len();
                let mut result = self.zeros(&a.shape)?;

                kernels.add(&a.data, &b.data, &mut result.data, n)?;
                self.synchronize()?;

                return Ok(result);
            }
        }

        // Fallback: not available without custom kernels
        Err(RustyGradientsError::BackendError(
            "CUDA add requires custom kernels. Compile PTX first.".to_string()
        ))
    }

    fn sub(&self, a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref kernels) = self.kernels {
                let n = a.len();
                let mut result = self.zeros(&a.shape)?;

                kernels.sub(&a.data, &b.data, &mut result.data, n)?;
                self.synchronize()?;

                return Ok(result);
            }
        }

        Err(RustyGradientsError::BackendError(
            "CUDA sub requires custom kernels".to_string()
        ))
    }

    fn mul(&self, a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref kernels) = self.kernels {
                let n = a.len();
                let mut result = self.zeros(&a.shape)?;

                kernels.mul(&a.data, &b.data, &mut result.data, n)?;
                self.synchronize()?;

                return Ok(result);
            }
        }

        Err(RustyGradientsError::BackendError(
            "CUDA mul requires custom kernels".to_string()
        ))
    }

    fn matmul(&self, a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage> {
        // Matrix multiplication using cuBLAS
        //
        // Expected performance: 5,000+ GFLOPS (vs 500 GFLOPS on CPU with BLAS)
        // = 10x speedup for large matrices

        // Validate shapes for matrix multiplication
        // a: [batch, m, k], b: [batch, k, n] → result: [batch, m, n]

        if a.shape.len() < 2 || b.shape.len() < 2 {
            return Err(RustyGradientsError::ShapeMismatch {
                expected: vec![2], // at least 2D
                actual: vec![a.shape.len(), b.shape.len()],
                context: "Matrix multiplication requires 2D+ tensors".to_string(),
            });
        }

        // For 2D matmul: a = [m, k], b = [k, n]
        let m = a.shape[a.shape.len() - 2];
        let k = a.shape[a.shape.len() - 1];
        let k2 = b.shape[b.shape.len() - 2];
        let n = b.shape[b.shape.len() - 1];

        if k != k2 {
            return Err(RustyGradientsError::ShapeMismatch {
                expected: vec![k],
                actual: vec![k2],
                context: format!("Matrix dimension mismatch: {}x{} @ {}x{}", m, k, k2, n),
            });
        }

        // Handle batched matmul later - for now just 2D
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(RustyGradientsError::BackendError(
                "Batched matmul not yet implemented in CUDA backend".to_string()
            ));
        }

        // Allocate output on GPU
        let result_shape = vec![m, n];
        let mut result = self.zeros(&result_shape)?;

        // cuBLAS gemm: C = alpha * A @ B + beta * C
        // We want: C = A @ B (so alpha=1, beta=0)
        unsafe {
            self.blas.gemm(
                // Transposition flags
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N, // A is not transposed
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N, // B is not transposed
                // Matrix dimensions (note: cuBLAS uses column-major, we use row-major)
                n as i32, // columns of B (and result)
                m as i32, // rows of A (and result)
                k as i32, // shared dimension
                // Scalars
                &1.0f32,  // alpha
                // Matrices
                b.data.device_ptr() as *const f32, n as i32, // B with leading dimension
                a.data.device_ptr() as *const f32, k as i32, // A with leading dimension
                &0.0f32,  // beta
                result.data.device_ptr_mut() as *mut f32, n as i32, // C with leading dimension
            ).map_err(|e| RustyGradientsError::BackendError(
                format!("cuBLAS gemm failed: {:?}", e)
            ))?;
        }

        Ok(result)
    }

    // === Element-wise Operations ===

    fn relu(&self, a: &Self::Storage) -> Result<Self::Storage> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref kernels) = self.kernels {
                let n = a.len();
                let mut result = self.zeros(&a.shape)?;

                kernels.relu(&a.data, &mut result.data, n)?;
                self.synchronize()?;

                return Ok(result);
            }
        }

        Err(RustyGradientsError::BackendError(
            "CUDA ReLU requires custom kernels".to_string()
        ))
    }

    fn sigmoid(&self, a: &Self::Storage) -> Result<Self::Storage> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref kernels) = self.kernels {
                let n = a.len();
                let mut result = self.zeros(&a.shape)?;

                kernels.sigmoid(&a.data, &mut result.data, n)?;
                self.synchronize()?;

                return Ok(result);
            }
        }

        Err(RustyGradientsError::BackendError(
            "CUDA sigmoid requires custom kernels".to_string()
        ))
    }

    fn exp(&self, a: &Self::Storage) -> Result<Self::Storage> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref kernels) = self.kernels {
                let n = a.len();
                let mut result = self.zeros(&a.shape)?;

                kernels.exp(&a.data, &mut result.data, n)?;
                self.synchronize()?;

                return Ok(result);
            }
        }

        Err(RustyGradientsError::BackendError(
            "CUDA exp requires custom kernels".to_string()
        ))
    }

    fn log(&self, a: &Self::Storage) -> Result<Self::Storage> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref kernels) = self.kernels {
                let n = a.len();
                let mut result = self.zeros(&a.shape)?;

                kernels.log(&a.data, &mut result.data, n)?;
                self.synchronize()?;

                return Ok(result);
            }
        }

        Err(RustyGradientsError::BackendError(
            "CUDA log requires custom kernels".to_string()
        ))
    }

    fn powf(&self, a: &Self::Storage, power: f32) -> Result<Self::Storage> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref kernels) = self.kernels {
                let n = a.len();
                let mut result = self.zeros(&a.shape)?;

                kernels.powf(&a.data, &mut result.data, power, n)?;
                self.synchronize()?;

                return Ok(result);
            }
        }

        Err(RustyGradientsError::BackendError(
            "CUDA powf requires custom kernels".to_string()
        ))
    }

    fn softmax(&self, a: &Self::Storage) -> Result<Self::Storage> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref kernels) = self.kernels {
                // Assume last dimension is the softmax axis
                // Shape: [batch, n] where we apply softmax over n
                if a.shape.len() < 2 {
                    return Err(RustyGradientsError::ShapeMismatch {
                        expected: vec![2],
                        actual: a.shape.clone(),
                        context: "Softmax requires at least 2D tensor".to_string(),
                    });
                }

                let batch: usize = a.shape.iter().take(a.shape.len() - 1).product();
                let n = a.shape[a.shape.len() - 1];
                let mut result = self.zeros(&a.shape)?;

                kernels.softmax(&a.data, &mut result.data, batch, n)?;
                self.synchronize()?;

                return Ok(result);
            }
        }

        Err(RustyGradientsError::BackendError(
            "CUDA softmax requires custom kernels".to_string()
        ))
    }

    // === Reduction Operations ===

    fn sum(&self, a: &Self::Storage) -> Result<Self::Storage> {
        unimplemented!("CUDA sum not yet implemented")
    }

    fn sum_axis(&self, a: &Self::Storage, axis: usize) -> Result<Self::Storage> {
        unimplemented!("CUDA sum_axis not yet implemented")
    }

    // === Transformation Operations ===

    fn reshape(&self, a: &Self::Storage, new_shape: &[usize]) -> Result<Self::Storage> {
        // Reshape is just a metadata operation (no data copy needed)
        let old_len: usize = a.shape.iter().product();
        let new_len: usize = new_shape.iter().product();

        if old_len != new_len {
            return Err(RustyGradientsError::ShapeMismatch {
                expected: vec![new_len],
                actual: vec![old_len],
                context: "Reshape: total elements must match".to_string(),
            });
        }

        Ok(CudaStorage {
            data: a.data.clone(),
            shape: new_shape.to_vec(),
            device: a.device.clone(),
        })
    }

    fn transpose(&self, a: &Self::Storage, axis1: usize, axis2: usize) -> Result<Self::Storage> {
        // TODO: Implement efficient CUDA transpose kernel
        unimplemented!("CUDA transpose not yet implemented")
    }

    // === Special Operations ===

    fn embedding(&self, indices: &Self::Storage, weights: &Self::Storage) -> Result<Self::Storage> {
        unimplemented!("CUDA embedding not yet implemented")
    }

    fn layer_norm(
        &self,
        x: &Self::Storage,
        gamma: &Self::Storage,
        beta: &Self::Storage,
        epsilon: f32,
    ) -> Result<Self::Storage> {
        // TODO: Implement fused LayerNorm kernel
        unimplemented!("CUDA layer_norm not yet implemented")
    }

    fn sparse_cross_entropy(
        &self,
        logits: &Self::Storage,
        targets: &Self::Storage,
    ) -> Result<Self::Storage> {
        unimplemented!("CUDA sparse_cross_entropy not yet implemented")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_available() {
        let count = CudaBackend::device_count();
        println!("CUDA devices available: {}", count);

        if count > 0 {
            println!("CUDA is available!");
        } else {
            println!("No CUDA devices found - tests will be skipped");
        }
    }

    #[test]
    #[ignore] // Only run if CUDA is available
    fn test_cuda_init() {
        if !CudaBackend::is_available() {
            return;
        }

        let backend = CudaBackend::new(0).expect("Failed to create CUDA backend");
        assert_eq!(backend.device_type(), DeviceType::Cuda(0));
    }

    #[test]
    #[ignore]
    fn test_cuda_memory() {
        if !CudaBackend::is_available() {
            return;
        }

        let backend = CudaBackend::new(0).unwrap();

        // Create zeros
        let zeros = backend.zeros(&[10, 20]).unwrap();
        assert_eq!(zeros.shape, vec![10, 20]);

        // Copy back to host
        let data = backend.to_vec(&zeros).unwrap();
        assert_eq!(data.len(), 200);
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    #[ignore]
    fn test_cuda_matmul() {
        if !CudaBackend::is_available() {
            return;
        }

        let backend = CudaBackend::new(0).unwrap();

        // Create test matrices
        let a_data = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b_data = vec![5.0, 6.0, 7.0, 8.0]; // 2x2

        let a = backend.from_slice(&a_data, &[2, 2]).unwrap();
        let b = backend.from_slice(&b_data, &[2, 2]).unwrap();

        // Matmul
        let c = backend.matmul(&a, &b).unwrap();

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[19, 22], [43, 50]]
        let result = backend.to_vec(&c).unwrap();

        assert_eq!(result.len(), 4);
        assert!((result[0] - 19.0).abs() < 1e-5);
        assert!((result[1] - 22.0).abs() < 1e-5);
        assert!((result[2] - 43.0).abs() < 1e-5);
        assert!((result[3] - 50.0).abs() < 1e-5);
    }
}
