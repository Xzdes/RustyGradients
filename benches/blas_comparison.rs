///! Benchmark comparing naive ndarray vs BLAS-accelerated matmul
///!
///! Run with:
///!   cargo bench --bench blas_comparison                    # Naive (no BLAS)
///!   cargo bench --bench blas_comparison --features cpu-blas # BLAS-optimized
///!
///! Expected results:
///!   Naive:     512×512 matmul ~100-500ms
///!   BLAS:      512×512 matmul ~2-10ms
///!   Speedup:   10-50x

use rusty_gradients::backend::cpu::CpuBackend;
use rusty_gradients::backend::Backend;
use rusty_gradients::tensor_v2::TensorV2;
use std::time::Instant;

fn benchmark_matmul(size: usize, iterations: usize) -> f64 {
    // Create random matrices using TensorV2
    let a = TensorV2::randn(&[size, size], false);
    let b = TensorV2::randn(&[size, size], false);

    // Get CPU data for direct backend access
    let backend = CpuBackend::new();
    let a_data = a.to_cpu_data().unwrap();
    let b_data = b.to_cpu_data().unwrap();

    // Warmup
    for _ in 0..3 {
        let _ = backend.matmul(&a_data, &b_data).unwrap();
    }

    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = backend.matmul(&a_data, &b_data).unwrap();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

fn benchmark_batched_matmul(batch_size: usize, m: usize, k: usize, n: usize, iterations: usize) -> f64 {
    let a = TensorV2::randn(&[batch_size, m, k], false);
    let b = TensorV2::randn(&[batch_size, k, n], false);

    let backend = CpuBackend::new();
    let a_data = a.to_cpu_data().unwrap();
    let b_data = b.to_cpu_data().unwrap();

    // Warmup
    for _ in 0..3 {
        let _ = backend.matmul(&a_data, &b_data).unwrap();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = backend.matmul(&a_data, &b_data).unwrap();
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() / iterations as f64
}

fn main() {
    println!("=== BLAS vs Naive Matmul Benchmark ===\n");

    #[cfg(feature = "cpu-blas")]
    println!("✓ Running with BLAS optimization (OpenBLAS)\n");

    #[cfg(not(feature = "cpu-blas"))]
    println!("✗ Running WITHOUT BLAS (naive ndarray)\n");

    // Small matrices
    println!("1. Small matrices (128×128):");
    let time = benchmark_matmul(128, 100);
    println!("   Average time: {:.3} ms", time * 1000.0);
    println!("   Throughput: {:.1} GFLOPS\n", (2.0 * 128f64.powi(3)) / time / 1e9);

    // Medium matrices
    println!("2. Medium matrices (512×512):");
    let time = benchmark_matmul(512, 20);
    println!("   Average time: {:.3} ms", time * 1000.0);
    println!("   Throughput: {:.1} GFLOPS\n", (2.0 * 512f64.powi(3)) / time / 1e9);

    // Large matrices
    println!("3. Large matrices (1024×1024):");
    let time = benchmark_matmul(1024, 5);
    println!("   Average time: {:.3} ms", time * 1000.0);
    println!("   Throughput: {:.1} GFLOPS\n", (2.0 * 1024f64.powi(3)) / time / 1e9);

    // Batched matmul (common in neural networks)
    println!("4. Batched matmul (32 × [256×256]):");
    let time = benchmark_batched_matmul(32, 256, 256, 256, 10);
    println!("   Average time: {:.3} ms", time * 1000.0);
    println!("   Throughput: {:.1} GFLOPS\n", (32.0 * 2.0 * 256f64.powi(3)) / time / 1e9);

    // Multi-head attention pattern
    println!("5. Multi-head attention (8 heads × 512 seq × 64 dim):");
    let time = benchmark_batched_matmul(8, 512, 64, 512, 10);
    println!("   Average time: {:.3} ms", time * 1000.0);
    println!("   Per-head: {:.3} ms\n", time * 1000.0 / 8.0);

    println!("=== Benchmark Complete ===");
    println!("\nTo compare BLAS vs naive:");
    println!("  1. Run: cargo bench --bench blas_comparison");
    println!("  2. Run: cargo bench --bench blas_comparison --features cpu-blas");
    println!("  3. Compare throughput (GFLOPS) - expect 10-50x improvement");
}
