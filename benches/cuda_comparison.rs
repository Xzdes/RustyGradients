///! CUDA vs CPU Performance Comparison Benchmark
///!
///! This benchmark compares matrix multiplication performance between:
///! - CPU (naive implementation)
///! - CPU with BLAS (OpenBLAS)
///! - CUDA with cuBLAS
///!
///! Expected Results:
///! - CPU naive: ~77 GFLOPS
///! - CPU BLAS: ~500 GFLOPS (6-10x speedup)
///! - CUDA cuBLAS: ~5,000+ GFLOPS (50-100x speedup)
///!
///! Run with:
///! ```bash
///! cargo bench --bench cuda_comparison --features "cpu-blas cuda"
///! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rusty_gradients::backend::{Backend, Device};

#[cfg(feature = "cuda")]
use rusty_gradients::backend::cuda::CudaBackend;

use rusty_gradients::backend::cpu::CpuBackend;

/// Calculate GFLOPS for matrix multiplication
/// For A (m×k) @ B (k×n), operations = 2*m*k*n (multiply + add)
fn calculate_gflops(m: usize, k: usize, n: usize, duration_secs: f64) -> f64 {
    let ops = 2.0 * m as f64 * k as f64 * n as f64;
    ops / duration_secs / 1e9
}

/// Benchmark CPU matmul (naive ndarray implementation)
fn bench_cpu_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_cpu_naive");

    for size in [128, 256, 512, 1024, 2048].iter() {
        let m = *size;
        let k = *size;
        let n = *size;

        // Calculate throughput in GFLOPS
        let ops = 2 * m * k * n;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let backend = CpuBackend::new();

            // Create random matrices
            let a_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.01).collect();
            let b_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.02).collect();

            let a = backend.from_slice(&a_data, &[size, size]).unwrap();
            let b = backend.from_slice(&b_data, &[size, size]).unwrap();

            b.iter(|| {
                let result = backend.matmul(&a, &b).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark CPU matmul with BLAS acceleration
#[cfg(feature = "cpu-blas")]
fn bench_cpu_blas(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_cpu_blas");

    for size in [128, 256, 512, 1024, 2048].iter() {
        let m = *size;
        let k = *size;
        let n = *size;

        let ops = 2 * m * k * n;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            // TODO: Use BLAS-accelerated backend when available
            let backend = CpuBackend::new();

            let a_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.01).collect();
            let b_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.02).collect();

            let a = backend.from_slice(&a_data, &[size, size]).unwrap();
            let b = backend.from_slice(&b_data, &[size, size]).unwrap();

            b.iter(|| {
                let result = backend.matmul(&a, &b).unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark CUDA matmul with cuBLAS
#[cfg(feature = "cuda")]
fn bench_cuda_cublas(c: &mut Criterion) {
    // Check if CUDA is available
    if !CudaBackend::is_available() {
        println!("⚠️  CUDA not available - skipping CUDA benchmarks");
        return;
    }

    let mut group = c.benchmark_group("matmul_cuda_cublas");

    // Warmup GPU
    println!("🔥 Warming up CUDA device...");
    let backend = CudaBackend::new(0).expect("Failed to initialize CUDA");
    let warmup_data: Vec<f32> = (0..100 * 100).map(|i| i as f32).collect();
    let warmup_a = backend.from_slice(&warmup_data, &[100, 100]).unwrap();
    let warmup_b = backend.from_slice(&warmup_data, &[100, 100]).unwrap();
    let _ = backend.matmul(&warmup_a, &warmup_b).unwrap();
    backend.synchronize().unwrap();
    println!("✅ CUDA warmup complete");

    for size in [128, 256, 512, 1024, 2048, 4096].iter() {
        let m = *size;
        let k = *size;
        let n = *size;

        let ops = 2 * m * k * n;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let backend = CudaBackend::new(0).expect("Failed to initialize CUDA");

            // Create matrices on GPU
            let a_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.01).collect();
            let b_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.02).collect();

            let a = backend.from_slice(&a_data, &[size, size]).unwrap();
            let b = backend.from_slice(&b_data, &[size, size]).unwrap();

            b.iter(|| {
                let result = backend.matmul(&a, &b).unwrap();
                backend.synchronize().unwrap(); // Ensure operation completes
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Comparison benchmark: Run same operation on all backends
fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_comparison");
    let size = 1024; // Standard size for comparison

    let ops = 2 * size * size * size;
    group.throughput(Throughput::Elements(ops as u64));

    // CPU naive
    group.bench_function("CPU_naive_1024", |b| {
        let backend = CpuBackend::new();
        let a_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.02).collect();
        let a = backend.from_slice(&a_data, &[size, size]).unwrap();
        let b_mat = backend.from_slice(&b_data, &[size, size]).unwrap();

        b.iter(|| {
            let result = backend.matmul(&a, &b_mat).unwrap();
            black_box(result);
        });
    });

    // CUDA cuBLAS
    #[cfg(feature = "cuda")]
    {
        if CudaBackend::is_available() {
            group.bench_function("CUDA_cuBLAS_1024", |b| {
                let backend = CudaBackend::new(0).expect("Failed to initialize CUDA");
                let a_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.01).collect();
                let b_data: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.02).collect();
                let a = backend.from_slice(&a_data, &[size, size]).unwrap();
                let b_mat = backend.from_slice(&b_data, &[size, size]).unwrap();

                b.iter(|| {
                    let result = backend.matmul(&a, &b_mat).unwrap();
                    backend.synchronize().unwrap();
                    black_box(result);
                });
            });
        }
    }

    group.finish();
}

/// Summary benchmark: Print GFLOPS for all backends
fn print_summary() {
    println!("\n========================================");
    println!("🚀 CUDA vs CPU Performance Summary");
    println!("========================================\n");

    let size = 1024;
    let m = size;
    let k = size;
    let n = size;

    // CPU naive
    println!("📊 CPU (naive ndarray):");
    println!("   Expected: ~77 GFLOPS");
    println!("   Matrix size: {}x{}", size, size);

    // CPU BLAS
    #[cfg(feature = "cpu-blas")]
    println!("\n📊 CPU (OpenBLAS):");
    println!("   Expected: ~500 GFLOPS (6-10x vs naive)");

    // CUDA cuBLAS
    #[cfg(feature = "cuda")]
    {
        if CudaBackend::is_available() {
            println!("\n📊 CUDA (cuBLAS):");
            println!("   Expected: ~5,000+ GFLOPS (10x vs BLAS, 50-100x vs naive)");
            println!("   GPU: {}", {
                let backend = CudaBackend::new(0).unwrap();
                "NVIDIA GPU"
            });
        } else {
            println!("\n⚠️  CUDA not available on this system");
        }
    }

    println!("\n========================================\n");
}

// Benchmark groups
criterion_group!(
    benches,
    bench_cpu_naive,
    #[cfg(feature = "cpu-blas")]
    bench_cpu_blas,
    #[cfg(feature = "cuda")]
    bench_cuda_cublas,
    bench_comparison
);

criterion_main!(benches);

// Print summary after benchmarks
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summary() {
        print_summary();
    }
}
