///! Benchmark comparing SIMD-optimized vs scalar elementwise operations
///!
///! Run with:
///!   cargo bench --bench simd_benchmark                    # Baseline (with rayon)
///!   cargo bench --bench simd_benchmark --features simd    # SIMD-optimized (future)
///!
///! Expected results:
///!   Rayon parallelization: 2-4x speedup (multi-core)
///!   SIMD vectorization:    4-8x speedup (single-core)
///!   Combined:              8-16x speedup (ideal)

use rusty_gradients::backend::cpu::CpuBackend;
use rusty_gradients::backend::Backend;
use rusty_gradients::tensor_v2::TensorV2;
use std::time::Instant;

fn benchmark_elementwise<F>(
    op_name: &str,
    size: usize,
    iterations: usize,
    op: F,
) -> f64
where
    F: Fn(&ndarray::ArrayD<f32>) -> rusty_gradients::error::Result<ndarray::ArrayD<f32>>,
{
    let tensor = TensorV2::randn(&[size], false);
    let backend = CpuBackend::new();
    let data = tensor.to_cpu_data().unwrap();

    // Warmup
    for _ in 0..3 {
        let _ = op(&data).unwrap();
    }

    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = op(&data).unwrap();
    }
    let elapsed = start.elapsed();

    let time_per_iter = elapsed.as_secs_f64() / iterations as f64;
    let throughput = (size as f64) / time_per_iter / 1e9; // GElements/s

    println!(
        "   {}: {:.3} ms/iter, {:.2} GElements/s",
        op_name,
        time_per_iter * 1000.0,
        throughput
    );

    time_per_iter
}

fn main() {
    println!("=== SIMD Elementwise Operations Benchmark ===\n");

    #[cfg(feature = "simd")]
    println!("✓ Running with SIMD optimization\n");

    #[cfg(all(not(feature = "simd"), feature = "cpu"))]
    println!("✓ Running with rayon parallelization (no SIMD)\n");

    #[cfg(not(any(feature = "simd", feature = "cpu")))]
    println!("✗ Running scalar baseline (no optimizations)\n");

    let backend = CpuBackend::new();

    // Small arrays (fit in L1 cache)
    println!("1. Small arrays (1M elements):");
    let size = 1_000_000;
    benchmark_elementwise("ReLU", size, 100, |a| backend.relu(a));
    benchmark_elementwise("Exp", size, 100, |a| backend.exp(a));
    benchmark_elementwise("Sigmoid", size, 100, |a| backend.sigmoid(a));
    benchmark_elementwise("Power(x^2)", size, 100, |a| backend.powf(a, 2.0));
    println!();

    // Medium arrays
    println!("2. Medium arrays (10M elements):");
    let size = 10_000_000;
    benchmark_elementwise("ReLU", size, 20, |a| backend.relu(a));
    benchmark_elementwise("Exp", size, 20, |a| backend.exp(a));
    benchmark_elementwise("Sigmoid", size, 20, |a| backend.sigmoid(a));
    println!();

    // Large arrays (memory-bound)
    println!("3. Large arrays (50M elements):");
    let size = 50_000_000;
    benchmark_elementwise("ReLU", size, 5, |a| backend.relu(a));
    benchmark_elementwise("Exp", size, 5, |a| backend.exp(a));
    println!();

    // Typical neural network activations
    println!("4. Typical NN layer (batch=32, seq=512, dim=768):");
    let batch_size = 32;
    let seq_len = 512;
    let hidden_dim = 768;
    let total_size = batch_size * seq_len * hidden_dim;

    let tensor = TensorV2::randn(&[batch_size, seq_len, hidden_dim], false);
    let data = tensor.to_cpu_data().unwrap();

    let start = Instant::now();
    for _ in 0..10 {
        let _ = backend.relu(&data).unwrap();
    }
    let elapsed = start.elapsed();
    println!(
        "   ReLU on [32, 512, 768]: {:.3} ms/iter",
        elapsed.as_secs_f64() / 10.0 * 1000.0
    );

    let start = Instant::now();
    for _ in 0..10 {
        let _ = backend.sigmoid(&data).unwrap();
    }
    let elapsed = start.elapsed();
    println!(
        "   Sigmoid on [32, 512, 768]: {:.3} ms/iter",
        elapsed.as_secs_f64() / 10.0 * 1000.0
    );
    println!();

    println!("=== Benchmark Complete ===");
    println!("\nExpected speedups:");
    println!("  Rayon (multi-core):   2-4x  (available with default cpu feature)");
    println!("  SIMD (single-core):   4-8x  (requires explicit SIMD implementation)");
    println!("  Combined:             8-16x (rayon + SIMD)");
    println!("\nCurrent implementation uses rayon parallelization.");
    println!("AVX2 SIMD path is partially implemented (ReLU only).");
}
