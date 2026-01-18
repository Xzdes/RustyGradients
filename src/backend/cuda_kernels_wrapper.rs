///! CUDA Kernels Wrapper
///!
///! Rust wrapper for custom CUDA kernels.
///! Loads PTX and provides safe Rust API.

use crate::error::{Result, RustyGradientsError};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

/// CUDA Kernels Manager
/// Loads and manages custom CUDA kernels
pub struct CudaKernels {
    device: Arc<CudaDevice>,

    // Elementwise kernels
    elementwise_add: CudaFunction,
    elementwise_mul: CudaFunction,
    elementwise_sub: CudaFunction,
    elementwise_div: CudaFunction,

    // Activation kernels
    relu_forward: CudaFunction,
    relu_backward: CudaFunction,
    sigmoid_forward: CudaFunction,
    gelu_forward: CudaFunction,
    exp_forward: CudaFunction,
    log_forward: CudaFunction,
    powf_forward: CudaFunction,

    // Reduction kernels
    sum_reduce: CudaFunction,

    // Advanced kernels
    softmax_forward: CudaFunction,
    layernorm_forward: CudaFunction,

    // Utility kernels
    fill_constant: CudaFunction,
    copy_tensor: CudaFunction,
    scalar_mul: CudaFunction,
    scalar_add: CudaFunction,
}

impl CudaKernels {
    /// Load CUDA kernels from PTX file
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        // Load PTX file (compiled by build.rs)
        let ptx_path = env!("CUDA_KERNELS_PTX");

        println!("📦 Loading CUDA kernels from: {}", ptx_path);

        let ptx = std::fs::read_to_string(ptx_path)
            .map_err(|e| RustyGradientsError::BackendError(
                format!("Failed to load CUDA kernels PTX: {}. Make sure to compile with --features cuda", e)
            ))?;

        // Load PTX module
        device.load_ptx(ptx.into(), "cuda_kernels", &[
            "elementwise_add",
            "elementwise_mul",
            "elementwise_sub",
            "elementwise_div",
            "relu_forward",
            "relu_backward",
            "sigmoid_forward",
            "gelu_forward",
            "exp_forward",
            "log_forward",
            "powf_forward",
            "sum_reduce",
            "softmax_forward",
            "layernorm_forward",
            "fill_constant",
            "copy_tensor",
            "scalar_mul",
            "scalar_add",
        ])
        .map_err(|e| RustyGradientsError::BackendError(
            format!("Failed to load CUDA PTX module: {:?}", e)
        ))?;

        println!("✅ CUDA kernels loaded successfully!");

        // Get kernel functions
        Ok(Self {
            elementwise_add: device.get_func("cuda_kernels", "elementwise_add").unwrap(),
            elementwise_mul: device.get_func("cuda_kernels", "elementwise_mul").unwrap(),
            elementwise_sub: device.get_func("cuda_kernels", "elementwise_sub").unwrap(),
            elementwise_div: device.get_func("cuda_kernels", "elementwise_div").unwrap(),
            relu_forward: device.get_func("cuda_kernels", "relu_forward").unwrap(),
            relu_backward: device.get_func("cuda_kernels", "relu_backward").unwrap(),
            sigmoid_forward: device.get_func("cuda_kernels", "sigmoid_forward").unwrap(),
            gelu_forward: device.get_func("cuda_kernels", "gelu_forward").unwrap(),
            exp_forward: device.get_func("cuda_kernels", "exp_forward").unwrap(),
            log_forward: device.get_func("cuda_kernels", "log_forward").unwrap(),
            powf_forward: device.get_func("cuda_kernels", "powf_forward").unwrap(),
            sum_reduce: device.get_func("cuda_kernels", "sum_reduce").unwrap(),
            softmax_forward: device.get_func("cuda_kernels", "softmax_forward").unwrap(),
            layernorm_forward: device.get_func("cuda_kernels", "layernorm_forward").unwrap(),
            fill_constant: device.get_func("cuda_kernels", "fill_constant").unwrap(),
            copy_tensor: device.get_func("cuda_kernels", "copy_tensor").unwrap(),
            scalar_mul: device.get_func("cuda_kernels", "scalar_mul").unwrap(),
            scalar_add: device.get_func("cuda_kernels", "scalar_add").unwrap(),
            device,
        })
    }

    /// Calculate optimal grid/block dimensions for 1D kernel
    fn get_launch_config_1d(&self, n: usize) -> LaunchConfig {
        let threads_per_block = 256;
        let blocks = (n + threads_per_block - 1) / threads_per_block;
        LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Calculate launch config with shared memory
    fn get_launch_config_with_shared(&self, batch: usize, shared_mem_bytes: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (batch as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes,
        }
    }

    // ========================================================================
    // Elementwise Operations
    // ========================================================================

    pub fn add(&self, a: &CudaSlice<f32>, b: &CudaSlice<f32>, c: &mut CudaSlice<f32>, n: usize) -> Result<()> {
        let config = self.get_launch_config_1d(n);
        unsafe {
            self.elementwise_add.launch(config, (a, b, c, n as i32))
                .map_err(|e| RustyGradientsError::BackendError(
                    format!("CUDA add kernel launch failed: {:?}", e)
                ))?;
        }
        Ok(())
    }

    pub fn mul(&self, a: &CudaSlice<f32>, b: &CudaSlice<f32>, c: &mut CudaSlice<f32>, n: usize) -> Result<()> {
        let config = self.get_launch_config_1d(n);
        unsafe {
            self.elementwise_mul.launch(config, (a, b, c, n as i32))
                .map_err(|e| RustyGradientsError::BackendError(
                    format!("CUDA mul kernel launch failed: {:?}", e)
                ))?;
        }
        Ok(())
    }

    pub fn sub(&self, a: &CudaSlice<f32>, b: &CudaSlice<f32>, c: &mut CudaSlice<f32>, n: usize) -> Result<()> {
        let config = self.get_launch_config_1d(n);
        unsafe {
            self.elementwise_sub.launch(config, (a, b, c, n as i32))
                .map_err(|e| RustyGradientsError::BackendError(
                    format!("CUDA sub kernel launch failed: {:?}", e)
                ))?;
        }
        Ok(())
    }

    // ========================================================================
    // Activation Functions
    // ========================================================================

    pub fn relu(&self, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, n: usize) -> Result<()> {
        let config = self.get_launch_config_1d(n);
        unsafe {
            self.relu_forward.launch(config, (x, y, n as i32))
                .map_err(|e| RustyGradientsError::BackendError(
                    format!("CUDA ReLU kernel launch failed: {:?}", e)
                ))?;
        }
        Ok(())
    }

    pub fn sigmoid(&self, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, n: usize) -> Result<()> {
        let config = self.get_launch_config_1d(n);
        unsafe {
            self.sigmoid_forward.launch(config, (x, y, n as i32))
                .map_err(|e| RustyGradientsError::BackendError(
                    format!("CUDA sigmoid kernel launch failed: {:?}", e)
                ))?;
        }
        Ok(())
    }

    pub fn exp(&self, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, n: usize) -> Result<()> {
        let config = self.get_launch_config_1d(n);
        unsafe {
            self.exp_forward.launch(config, (x, y, n as i32))
                .map_err(|e| RustyGradientsError::BackendError(
                    format!("CUDA exp kernel launch failed: {:?}", e)
                ))?;
        }
        Ok(())
    }

    pub fn log(&self, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, n: usize) -> Result<()> {
        let config = self.get_launch_config_1d(n);
        unsafe {
            self.log_forward.launch(config, (x, y, n as i32))
                .map_err(|e| RustyGradientsError::BackendError(
                    format!("CUDA log kernel launch failed: {:?}", e)
                ))?;
        }
        Ok(())
    }

    pub fn powf(&self, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, power: f32, n: usize) -> Result<()> {
        let config = self.get_launch_config_1d(n);
        unsafe {
            self.powf_forward.launch(config, (x, y, power, n as i32))
                .map_err(|e| RustyGradientsError::BackendError(
                    format!("CUDA powf kernel launch failed: {:?}", e)
                ))?;
        }
        Ok(())
    }

    // ========================================================================
    // Advanced Kernels
    // ========================================================================

    pub fn softmax(&self, input: &CudaSlice<f32>, output: &mut CudaSlice<f32>, batch: usize, n: usize) -> Result<()> {
        let shared_mem_bytes = 512 * std::mem::size_of::<f32>() as u32; // 2x 256 floats
        let config = self.get_launch_config_with_shared(batch, shared_mem_bytes);

        unsafe {
            self.softmax_forward.launch(config, (input, output, batch as i32, n as i32))
                .map_err(|e| RustyGradientsError::BackendError(
                    format!("CUDA softmax kernel launch failed: {:?}", e)
                ))?;
        }
        Ok(())
    }

    pub fn layernorm(
        &self,
        input: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        batch: usize,
        n: usize,
        epsilon: f32,
    ) -> Result<()> {
        let shared_mem_bytes = 512 * std::mem::size_of::<f32>() as u32;
        let config = self.get_launch_config_with_shared(batch, shared_mem_bytes);

        unsafe {
            self.layernorm_forward.launch(config, (input, gamma, beta, output, batch as i32, n as i32, epsilon))
                .map_err(|e| RustyGradientsError::BackendError(
                    format!("CUDA layernorm kernel launch failed: {:?}", e)
                ))?;
        }
        Ok(())
    }
}
