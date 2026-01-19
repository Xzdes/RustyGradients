///! Fused operations для CPU backend
///!
///! Оптимизированные реализации, которые объединяют несколько операций
///! в один проход для уменьшения memory traffic и повышения производительности.

use crate::error::{Result, RustyGradientsError};
use ndarray::{ArrayD, Axis, IxDyn};

/// Fused LayerNorm using Welford's one-pass algorithm
///
/// Вычисляет mean и variance за один проход по данным вместо двух.
/// Expected speedup: 2-4x over naive two-pass implementation.
///
/// Algorithm:
/// ```text
/// For each element x_i:
///   delta = x_i - mean_old
///   mean_new = mean_old + delta / (i + 1)
///   M2 = M2 + delta * (x_i - mean_new)
/// variance = M2 / n
/// ```
pub fn layer_norm_fused(
    x: &ArrayD<f32>,
    gamma: &ArrayD<f32>,
    beta: &ArrayD<f32>,
    epsilon: f32,
) -> Result<ArrayD<f32>> {
    let _last_axis = Axis(x.ndim() - 1);
    let normalized_dim = x.shape()[x.ndim() - 1];

    // Output shape same as input
    #[allow(unused_mut, unused_assignments)]
    let mut output = ArrayD::zeros(x.raw_dim());

    // Prepare shapes for broadcasting
    let mut param_shape = x.shape().to_vec();
    param_shape[x.ndim() - 1] = 1;

    // Process each normalization slice
    let num_slices = x.len() / normalized_dim;

    #[cfg(feature = "cpu")]
    {
        // Parallel processing of normalization slices
        use rayon::prelude::*;

        // Check if array is contiguous
        if x.as_slice().is_none() {
            // Fallback to naive implementation for non-contiguous arrays
            // TODO: Implement strided version
            return layer_norm_naive_fallback(x, gamma, beta, epsilon);
        }

        // Flatten input and output for parallel iteration
        let x_slice = x.as_slice().unwrap();
        let input_slices: Vec<_> = (0..num_slices)
            .map(|i| {
                let start = i * normalized_dim;
                let end = start + normalized_dim;
                &x_slice[start..end]
            })
            .collect();

        let output_vecs: Vec<Vec<f32>> = input_slices
            .par_iter()
            .enumerate()
            .map(|(_slice_idx, &slice)| {
                // Welford's one-pass algorithm
                let mut mean = 0.0f32;
                let mut m2 = 0.0f32;
                let n = slice.len() as f32;

                for (i, &value) in slice.iter().enumerate() {
                    let delta = value - mean;
                    mean += delta / (i + 1) as f32;
                    let delta2 = value - mean;
                    m2 += delta * delta2;
                }

                let variance = m2 / n;
                let std_inv = 1.0 / (variance + epsilon).sqrt();

                // Normalize and apply affine transformation
                let gamma_slice = gamma.as_slice().unwrap();
                let beta_slice = beta.as_slice().unwrap();

                slice
                    .iter()
                    .enumerate()
                    .map(|(i, &x_val)| {
                        let normalized = (x_val - mean) * std_inv;
                        gamma_slice[i] * normalized + beta_slice[i]
                    })
                    .collect()
            })
            .collect();

        // Flatten results back into output
        let mut output_data = Vec::with_capacity(x.len());
        for vec in output_vecs {
            output_data.extend(vec);
        }

        output = ArrayD::from_shape_vec(x.raw_dim(), output_data)
            .map_err(|e| RustyGradientsError::ShapeError(e.to_string()))?;
    }

    #[cfg(not(feature = "cpu"))]
    {
        // Sequential fallback
        for slice_idx in 0..num_slices {
            let start = slice_idx * normalized_dim;
            let end = start + normalized_dim;
            let slice = &x.as_slice().unwrap()[start..end];

            // Welford's algorithm
            let mut mean = 0.0f32;
            let mut m2 = 0.0f32;
            let n = slice.len() as f32;

            for (i, &value) in slice.iter().enumerate() {
                let delta = value - mean;
                mean += delta / (i + 1) as f32;
                let delta2 = value - mean;
                m2 += delta * delta2;
            }

            let variance = m2 / n;
            let std_inv = 1.0 / (variance + epsilon).sqrt();

            // Normalize and apply affine
            let gamma_slice = gamma.as_slice().unwrap();
            let beta_slice = beta.as_slice().unwrap();
            let output_slice = &mut output.as_slice_mut().unwrap()[start..end];

            for (i, &x_val) in slice.iter().enumerate() {
                let normalized = (x_val - mean) * std_inv;
                output_slice[i] = gamma_slice[i] * normalized + beta_slice[i];
            }
        }
    }

    Ok(output)
}

/// Fused GELU activation
///
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
///
/// Объединяет все операции в один проход для уменьшения memory allocations.
pub fn gelu_fused(x: &ArrayD<f32>) -> Result<ArrayD<f32>> {
    const SQRT_2_OVER_PI: f32 = 0.7978845608; // sqrt(2/π)
    const COEFF: f32 = 0.044715;

    #[cfg(feature = "cpu")]
    {
        use rayon::prelude::*;
        if let Some(slice) = x.as_slice() {
            let data: Vec<f32> = slice
                .par_iter()
                .map(|&val| {
                    let inner = SQRT_2_OVER_PI * (val + COEFF * val.powi(3));
                    0.5 * val * (1.0 + inner.tanh())
                })
                .collect();
            return Ok(ArrayD::from_shape_vec(x.raw_dim(), data).unwrap());
        }
    }

    // Fallback
    Ok(x.mapv(|val| {
        let inner = SQRT_2_OVER_PI * (val + COEFF * val.powi(3));
        0.5 * val * (1.0 + inner.tanh())
    }))
}

/// Naive fallback for non-contiguous arrays
fn layer_norm_naive_fallback(
    x: &ArrayD<f32>,
    gamma: &ArrayD<f32>,
    beta: &ArrayD<f32>,
    epsilon: f32,
) -> Result<ArrayD<f32>> {
    let last_axis = Axis(x.ndim() - 1);

    // Two-pass algorithm
    let mean = x
        .mean_axis(last_axis)
        .ok_or_else(|| RustyGradientsError::ShapeError("Failed to compute mean".to_string()))?;

    let mut mean_shape = x.shape().to_vec();
    mean_shape[x.ndim() - 1] = 1;
    let mean_reshaped = mean
        .into_shape_with_order(IxDyn(&mean_shape))
        .map_err(|e| RustyGradientsError::ShapeError(e.to_string()))?;

    let x_minus_mean = x - &mean_reshaped;

    let variance = x_minus_mean
        .mapv(|v| v.powi(2))
        .mean_axis(last_axis)
        .ok_or_else(|| RustyGradientsError::ShapeError("Failed to compute variance".to_string()))?;

    let variance_reshaped = variance
        .into_shape_with_order(IxDyn(&mean_shape))
        .map_err(|e| RustyGradientsError::ShapeError(e.to_string()))?;

    let std_dev_inv = (&variance_reshaped + epsilon).mapv(|v| 1.0 / v.sqrt());
    let x_normalized = &x_minus_mean * &std_dev_inv;

    let gamma_reshaped = gamma
        .clone()
        .into_shape_with_order(IxDyn(&mean_shape))
        .map_err(|e| RustyGradientsError::ShapeError(e.to_string()))?;
    let beta_reshaped = beta
        .clone()
        .into_shape_with_order(IxDyn(&mean_shape))
        .map_err(|e| RustyGradientsError::ShapeError(e.to_string()))?;

    Ok(&x_normalized * &gamma_reshaped + &beta_reshaped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_fused() {
        // Simple test: normalize [1, 2, 3, 4]
        let x = ArrayD::from_shape_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let gamma = ArrayD::from_shape_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let beta = ArrayD::from_shape_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).unwrap();

        let result = layer_norm_fused(&x, &gamma, &beta, 1e-5).unwrap();

        // Mean = 2.5, Variance = 1.25
        // Expected: normalized values around [-1.34, -0.45, 0.45, 1.34]
        let mean = result.mean().unwrap();
        assert!((mean - 0.0).abs() < 1e-3, "Mean should be ~0 after normalization");

        // Check variance is ~1
        let variance = result.mapv(|v| v.powi(2)).mean().unwrap();
        assert!(
            (variance - 1.0).abs() < 0.1,
            "Variance should be ~1 after normalization"
        );
    }

    #[test]
    fn test_gelu_fused() {
        let x = ArrayD::from_shape_vec(vec![3], vec![-1.0, 0.0, 1.0]).unwrap();
        let result = gelu_fused(&x).unwrap();

        // GELU(0) ≈ 0
        assert!((result[[1]] - 0.0).abs() < 0.01);

        // GELU is approximately identity for positive values
        assert!(result[[2]] > 0.8 && result[[2]] < 0.9);
    }
}
