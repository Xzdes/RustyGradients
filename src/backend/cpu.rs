///! CPU Backend implementation
///!
///! Реализация Backend trait для CPU с использованием ndarray и rayon для параллелизма.

use super::{Backend, DeviceType};
use crate::error::{Result, RustyGradientsError};
use ndarray::{Array, ArrayD, Axis, IxDyn};

#[cfg(feature = "cpu")]
use rayon::prelude::*;

/// CPU Backend - реализация на основе ndarray
/// CPU всегда доступен, дополнительного состояния не требуется
pub struct CpuBackend {}

impl CpuBackend {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    type Storage = ArrayD<f32>;

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn synchronize(&self) -> Result<()> {
        // CPU - синхронные операции, ничего делать не нужно
        Ok(())
    }

    // === Memory Operations ===

    fn zeros(&self, shape: &[usize]) -> Result<Self::Storage> {
        Ok(ArrayD::zeros(IxDyn(shape)))
    }

    fn ones(&self, shape: &[usize]) -> Result<Self::Storage> {
        Ok(ArrayD::ones(IxDyn(shape)))
    }

    fn from_slice(&self, data: &[f32], shape: &[usize]) -> Result<Self::Storage> {
        ArrayD::from_shape_vec(IxDyn(shape), data.to_vec())
            .map_err(|e| RustyGradientsError::ShapeError(format!("Failed to create array from slice: {}", e)))
    }

    fn to_vec(&self, storage: &Self::Storage) -> Result<Vec<f32>> {
        Ok(storage.iter().copied().collect())
    }

    fn shape(&self, storage: &Self::Storage) -> Vec<usize> {
        storage.shape().to_vec()
    }

    // === Arithmetic Operations ===

    fn add(&self, a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage> {
        // ndarray поддерживает broadcasting автоматически
        Ok(a + b)
    }

    fn sub(&self, a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage> {
        Ok(a - b)
    }

    fn mul(&self, a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage> {
        Ok(a * b)
    }

    fn matmul(&self, a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage> {
        // Используем существующую реализацию из ops::matmul
        // TODO: Оптимизировать с BLAS и rayon parallelization
        use ndarray::{s, Ix2};

        match (a.ndim(), b.ndim()) {
            (2, 2) => {
                let a_2d = a.view().into_dimensionality::<Ix2>()
                    .map_err(|e| RustyGradientsError::ShapeError(e.to_string()))?;
                let b_2d = b.view().into_dimensionality::<Ix2>()
                    .map_err(|e| RustyGradientsError::ShapeError(e.to_string()))?;

                Ok(a_2d.dot(&b_2d).into_dyn())
            }
            (3, 3) => {
                // Batched matmul для 3D тензоров
                let batch_size = a.shape()[0];
                let m = a.shape()[1];
                let k = a.shape()[2];
                let n = b.shape()[2];

                if b.shape() != &[batch_size, k, n] {
                    return Err(RustyGradientsError::ShapeError(format!(
                        "Incompatible shapes for 3D matmul: {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }

                let mut out = ArrayD::zeros(vec![batch_size, m, n]);

                #[cfg(feature = "cpu")]
                {
                    // Параллельная обработка batch dimension
                    use rayon::prelude::*;
                    let results: Vec<_> = (0..batch_size)
                        .into_par_iter()
                        .map(|b_idx| {
                            let a_mat = a.slice(s![b_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                            let b_mat = b.slice(s![b_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                            a_mat.dot(&b_mat)
                        })
                        .collect();

                    for (b_idx, result) in results.into_iter().enumerate() {
                        out.slice_mut(s![b_idx, .., ..]).assign(&result);
                    }
                }

                #[cfg(not(feature = "cpu"))]
                {
                    // Sequential fallback
                    for b_idx in 0..batch_size {
                        let a_mat = a.slice(s![b_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                        let b_mat = b.slice(s![b_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                        let mut c_mat = out.slice_mut(s![b_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                        c_mat.assign(&a_mat.dot(&b_mat));
                    }
                }

                Ok(out)
            }
            (4, 4) => {
                // 4D batched matmul (для multi-head attention)
                let batch_size = a.shape()[0];
                let heads = a.shape()[1];
                let m = a.shape()[2];
                let k = a.shape()[3];
                let n = b.shape()[3];

                if b.shape() != &[batch_size, heads, k, n] {
                    return Err(RustyGradientsError::ShapeError(format!(
                        "Incompatible shapes for 4D matmul: {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }

                let mut out = ArrayD::zeros(vec![batch_size, heads, m, n]);

                #[cfg(feature = "cpu")]
                {
                    // Параллелизация по batch и heads
                    use rayon::prelude::*;
                    let total_batches = batch_size * heads;
                    let results: Vec<_> = (0..total_batches)
                        .into_par_iter()
                        .map(|idx| {
                            let b_idx = idx / heads;
                            let h_idx = idx % heads;
                            let a_mat = a.slice(s![b_idx, h_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                            let b_mat = b.slice(s![b_idx, h_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                            (b_idx, h_idx, a_mat.dot(&b_mat))
                        })
                        .collect();

                    for (b_idx, h_idx, result) in results {
                        out.slice_mut(s![b_idx, h_idx, .., ..]).assign(&result);
                    }
                }

                #[cfg(not(feature = "cpu"))]
                {
                    // Sequential fallback
                    for b_idx in 0..batch_size {
                        for h_idx in 0..heads {
                            let a_mat = a.slice(s![b_idx, h_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                            let b_mat = b.slice(s![b_idx, h_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                            let mut c_mat = out.slice_mut(s![b_idx, h_idx, .., ..]).into_dimensionality::<Ix2>().unwrap();
                            c_mat.assign(&a_mat.dot(&b_mat));
                        }
                    }
                }

                Ok(out)
            }
            _ => Err(RustyGradientsError::ShapeError(format!(
                "Unsupported matmul dimensions: {}D and {}D",
                a.ndim(),
                b.ndim()
            ))),
        }
    }

    // === Element-wise Operations ===

    fn relu(&self, a: &Self::Storage) -> Result<Self::Storage> {
        Ok(a.mapv(|x| x.max(0.0)))
    }

    fn sigmoid(&self, a: &Self::Storage) -> Result<Self::Storage> {
        Ok(a.mapv(|x| 1.0 / (1.0 + (-x).exp())))
    }

    fn exp(&self, a: &Self::Storage) -> Result<Self::Storage> {
        Ok(a.mapv(|x| x.exp()))
    }

    fn log(&self, a: &Self::Storage) -> Result<Self::Storage> {
        Ok(a.mapv(|x| x.ln()))
    }

    fn powf(&self, a: &Self::Storage, power: f32) -> Result<Self::Storage> {
        Ok(a.mapv(|x| x.powf(power)))
    }

    fn softmax(&self, a: &Self::Storage) -> Result<Self::Storage> {
        let last_axis = Axis(a.ndim() - 1);

        // Numerically stable softmax: exp(x - max(x))
        let max_vals = a.map_axis(last_axis, |row| {
            row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v))
        });

        let max_reshaped = {
            let mut new_shape = a.shape().to_vec();
            new_shape[a.ndim() - 1] = 1;
            max_vals.into_shape(new_shape).unwrap()
        };

        let exp_vals = (a - &max_reshaped).mapv(|x| x.exp());

        let sum_exp = exp_vals.sum_axis(last_axis);
        let sum_reshaped = {
            let mut new_shape = a.shape().to_vec();
            new_shape[a.ndim() - 1] = 1;
            sum_exp.into_shape(new_shape).unwrap()
        };

        Ok(&exp_vals / &sum_reshaped)
    }

    // === Reduction Operations ===

    fn sum(&self, a: &Self::Storage) -> Result<Self::Storage> {
        let sum_val = a.sum();
        Ok(Array::from_elem(IxDyn(&[1]), sum_val))
    }

    fn sum_axis(&self, a: &Self::Storage, axis: usize) -> Result<Self::Storage> {
        if axis >= a.ndim() {
            return Err(RustyGradientsError::ShapeError(format!(
                "Axis {} out of bounds for array of dimension {}",
                axis,
                a.ndim()
            )));
        }
        Ok(a.sum_axis(Axis(axis)).into_dyn())
    }

    // === Transformation Operations ===

    fn reshape(&self, a: &Self::Storage, new_shape: &[usize]) -> Result<Self::Storage> {
        a.clone()
            .into_shape(IxDyn(new_shape))
            .map_err(|e| RustyGradientsError::ShapeError(format!("Reshape failed: {}", e)))
    }

    fn transpose(&self, a: &Self::Storage, axis1: usize, axis2: usize) -> Result<Self::Storage> {
        if axis1 >= a.ndim() || axis2 >= a.ndim() {
            return Err(RustyGradientsError::ShapeError(format!(
                "Transpose axes ({}, {}) out of bounds for {}D array",
                axis1,
                axis2,
                a.ndim()
            )));
        }

        let mut permutation: Vec<usize> = (0..a.ndim()).collect();
        permutation.swap(axis1, axis2);
        Ok(a.view().permuted_axes(permutation).to_owned())
    }

    // === Special Operations ===

    fn embedding(&self, indices: &Self::Storage, weights: &Self::Storage) -> Result<Self::Storage> {
        // indices: [batch_size, seq_len] или [batch_size * seq_len]
        // weights: [vocab_size, embedding_dim]

        if weights.ndim() != 2 {
            return Err(RustyGradientsError::ShapeError(format!(
                "Embedding weights must be 2D, got {}D",
                weights.ndim()
            )));
        }

        let vocab_size = weights.shape()[0];
        let embedding_dim = weights.shape()[1];

        let ids_flat: Vec<usize> = indices
            .iter()
            .map(|&x| x as usize)
            .collect();

        let batch_size = indices.shape()[0];
        let seq_len = if indices.ndim() == 2 {
            indices.shape()[1]
        } else {
            1
        };

        let mut result_data = Vec::with_capacity(batch_size * seq_len * embedding_dim);

        for &id in &ids_flat {
            if id >= vocab_size {
                return Err(RustyGradientsError::ShapeError(format!(
                    "Index {} out of bounds for vocabulary size {}",
                    id,
                    vocab_size
                )));
            }
            let embedding_vector = weights.slice(ndarray::s![id, ..]);
            result_data.extend_from_slice(embedding_vector.as_slice().unwrap());
        }

        let result_shape = if indices.ndim() == 2 {
            vec![batch_size, seq_len, embedding_dim]
        } else {
            vec![batch_size, embedding_dim]
        };

        ArrayD::from_shape_vec(IxDyn(&result_shape), result_data)
            .map_err(|e| RustyGradientsError::ShapeError(format!("Embedding failed: {}", e)))
    }

    fn layer_norm(
        &self,
        x: &Self::Storage,
        gamma: &Self::Storage,
        beta: &Self::Storage,
        epsilon: f32,
    ) -> Result<Self::Storage> {
        let last_axis = Axis(x.ndim() - 1);
        let normalized_shape = x.shape()[x.ndim() - 1];

        // Compute mean along last axis
        let mean = x.mean_axis(last_axis).ok_or_else(|| {
            RustyGradientsError::ShapeError("Failed to compute mean".to_string())
        })?;

        // Reshape mean for broadcasting
        let mut mean_shape = x.shape().to_vec();
        mean_shape[x.ndim() - 1] = 1;
        let mean_reshaped = mean.into_shape(IxDyn(&mean_shape))
            .map_err(|e| RustyGradientsError::ShapeError(e.to_string()))?;

        // x - mean
        let x_minus_mean = x - &mean_reshaped;

        // Compute variance
        let variance = x_minus_mean
            .mapv(|v| v.powi(2))
            .mean_axis(last_axis)
            .ok_or_else(|| RustyGradientsError::ShapeError("Failed to compute variance".to_string()))?;

        let variance_reshaped = variance.into_shape(IxDyn(&mean_shape))
            .map_err(|e| RustyGradientsError::ShapeError(e.to_string()))?;

        // 1 / sqrt(variance + epsilon)
        let std_dev_inv = (&variance_reshaped + epsilon).mapv(|v| 1.0 / v.sqrt());

        // Normalize
        let x_normalized = &x_minus_mean * &std_dev_inv;

        // Apply affine transformation: gamma * x_normalized + beta
        let gamma_reshaped = gamma.clone().into_shape(IxDyn(&mean_shape))
            .map_err(|e| RustyGradientsError::ShapeError(e.to_string()))?;
        let beta_reshaped = beta.clone().into_shape(IxDyn(&mean_shape))
            .map_err(|e| RustyGradientsError::ShapeError(e.to_string()))?;

        Ok(&x_normalized * &gamma_reshaped + &beta_reshaped)
    }

    fn sparse_cross_entropy(
        &self,
        logits: &Self::Storage,
        targets: &Self::Storage,
    ) -> Result<Self::Storage> {
        // logits: [batch_size, vocab_size] или [batch_size * seq_len, vocab_size]
        // targets: [batch_size] или [batch_size * seq_len]

        if logits.ndim() != 2 {
            return Err(RustyGradientsError::ShapeError(format!(
                "Logits must be 2D, got {}D",
                logits.ndim()
            )));
        }

        let batch_size = logits.shape()[0];
        let num_classes = logits.shape()[1];

        // Softmax для numerical stability
        let last_axis = Axis(1);
        let max_logits = logits.map_axis(last_axis, |row| {
            row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v))
        });

        let max_reshaped = max_logits.insert_axis(Axis(1));
        let exp_logits = (logits - &max_reshaped).mapv(|x| x.exp());
        let sum_exp = exp_logits.sum_axis(last_axis);
        let sum_reshaped = sum_exp.insert_axis(Axis(1));
        let probs = &exp_logits / &sum_reshaped;

        // Extract target probabilities and compute -log(p)
        let mut losses = Vec::with_capacity(batch_size);
        let targets_flat: Vec<usize> = targets.iter().map(|&x| x as usize).collect();

        for (i, &target_idx) in targets_flat.iter().enumerate() {
            if target_idx >= num_classes {
                return Err(RustyGradientsError::ShapeError(format!(
                    "Target index {} out of bounds for {} classes",
                    target_idx,
                    num_classes
                )));
            }
            let prob = probs[[i, target_idx]];
            losses.push(-(prob.ln()));
        }

        // Return mean loss as scalar
        let mean_loss = losses.iter().sum::<f32>() / batch_size as f32;
        Ok(Array::from_elem(IxDyn(&[1]), mean_loss))
    }
}
