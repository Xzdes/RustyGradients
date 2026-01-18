///! CUDA Custom Kernels Benchmark
///!
///! Benchmarks custom CUDA kernels vs CPU implementations:
///! - Elementwise operations (add, mul, relu, sigmoid, exp, log)
///! - Softmax (fused kernel)
///! - LayerNorm (fused kernel)
///!
///! Expected Results:
///! - Elementwise ops: 20-50x speedup vs CPU
///! - Softmax: 10-20x speedup vs CPU
///! - LayerNorm: 5-10x speedup vs CPU
///!
///! Run with:
///! ```bash
///! cargo bench --bench cuda_kernels_bench --features cuda
///! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(feature = "cuda")]
use rusty_gradients::backend::{Backend, cuda::CudaBackend};

use rusty_gradients::backend::cpu::CpuBackend;

/// Benchmark elementwise add: CPU vs CUDA
fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_add");

    for size in [1024, 4096, 16384, 65536, 262144, 1048576].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        // CPU benchmark
        group.bench_with_input(BenchmarkId::new("CPU", size), size, |b, &size| {
            let backend = CpuBackend::new();
            let a_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
            let b_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.02).collect();

            let a = backend.from_slice(&a_data, &[size]).unwrap();
            let b = backend.from_slice(&b_data, &[size]).unwrap();

            b.iter(|| {
                let result = backend.add(&a, &b).unwrap();
                black_box(result);
            });
        });

        // CUDA benchmark
        #[cfg(feature = "cuda")]
        {
            if CudaBackend::is_available() {
                group.bench_with_input(BenchmarkId::new("CUDA", size), size, |b, &size| {
                    let backend = CudaBackend::new(0).expect("Failed to init CUDA");
                    let a_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
                    let b_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.02).collect();

                    let a = backend.from_slice(&a_data, &[size]).unwrap();
                    let b_mat = backend.from_slice(&b_data, &[size]).unwrap();

                    b.iter(|| {
                        let result = backend.add(&a, &b_mat).unwrap();
                        backend.synchronize().unwrap();
                        black_box(result);
                    });
                });
            }
        }
    }

    group.finish();
}

/// Benchmark ReLU: CPU vs CUDA
fn bench_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu");

    for size in [1024, 4096, 16384, 65536, 262144, 1048576].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        // CPU benchmark
        group.bench_with_input(BenchmarkId::new("CPU", size), size, |b, &size| {
            let backend = CpuBackend::new();
            let a_data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01) - 0.5).collect();
            let a = backend.from_slice(&a_data, &[size]).unwrap();

            b.iter(|| {
                let result = backend.relu(&a).unwrap();
                black_box(result);
            });
        });

        // CUDA benchmark
        #[cfg(feature = "cuda")]
        {
            if CudaBackend::is_available() {
                group.bench_with_input(BenchmarkId::new("CUDA", size), size, |b, &size| {
                    let backend = CudaBackend::new(0).expect("Failed to init CUDA");
                    let a_data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01) - 0.5).collect();
                    let a = backend.from_slice(&a_data, &[size]).unwrap();

                    b.iter(|| {
                        let result = backend.relu(&a).unwrap();
                        backend.synchronize().unwrap();
                        black_box(result);
                    });
                });
            }
        }
    }

    group.finish();
}

/// Benchmark Softmax: CPU vs CUDA
fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    for size in [128, 256, 512, 1024, 2048].iter() {
        let batch = 64;
        let n = *size;
        let total_elements = batch * n;

        group.throughput(Throughput::Elements(total_elements as u64));

        // CPU benchmark
        group.bench_with_input(BenchmarkId::new("CPU", size), size, |b, &size| {
            let backend = CpuBackend::new();
            let a_data: Vec<f32> = (0..batch * size).map(|i| (i as f32) * 0.01).collect();
            let a = backend.from_slice(&a_data, &[batch, size]).unwrap();

            b.iter(|| {
                let result = backend.softmax(&a).unwrap();
                black_box(result);
            });
        });

        // CUDA benchmark
        #[cfg(feature = "cuda")]
        {
            if CudaBackend::is_available() {
                group.bench_with_input(BenchmarkId::new("CUDA", size), size, |b, &size| {
                    let backend = CudaBackend::new(0).expect("Failed to init CUDA");
                    let a_data: Vec<f32> = (0..batch * size).map(|i| (i as f32) * 0.01).collect();
                    let a = backend.from_slice(&a_data, &[batch, size]).unwrap();

                    b.iter(|| {
                        let result = backend.softmax(&a).unwrap();
                        backend.synchronize().unwrap();
                        black_box(result);
                    });
                });
            }
        }
    }

    group.finish();
}

/// Benchmark comparison summary
fn bench_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_comparison");

    let size = 262144; // 256K elements
    group.throughput(Throughput::Elements(size as u64));

    // Operations to benchmark
    let operations = vec![
        ("add", "Elementwise Addition"),
        ("mul", "Elementwise Multiplication"),
        ("relu", "ReLU Activation"),
        ("exp", "Exponential"),
    ];

    for (op_name, op_desc) in operations {
        // CPU benchmark
        group.bench_function(&format!("CPU_{}", op_name), |b| {
            let backend = CpuBackend::new();
            let a_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
            let b_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.02).collect();

            let a = backend.from_slice(&a_data, &[size]).unwrap();
            let b_mat = backend.from_slice(&b_data, &[size]).unwrap();

            b.iter(|| {
                let result = match op_name {
                    "add" => backend.add(&a, &b_mat).unwrap(),
                    "mul" => backend.mul(&a, &b_mat).unwrap(),
                    "relu" => backend.relu(&a).unwrap(),
                    "exp" => backend.exp(&a).unwrap(),
                    _ => unreachable!(),
                };
                black_box(result);
            });
        });

        // CUDA benchmark
        #[cfg(feature = "cuda")]
        {
            if CudaBackend::is_available() {
                group.bench_function(&format!("CUDA_{}", op_name), |b| {
                    let backend = CudaBackend::new(0).expect("Failed to init CUDA");
                    let a_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
                    let b_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.02).collect();

                    let a = backend.from_slice(&a_data, &[size]).unwrap();
                    let b_mat = backend.from_slice(&b_data, &[size]).unwrap();

                    b.iter(|| {
                        let result = match op_name {
                            "add" => backend.add(&a, &b_mat).unwrap(),
                            "mul" => backend.mul(&a, &b_mat).unwrap(),
                            "relu" => backend.relu(&a).unwrap(),
                            "exp" => backend.exp(&a).unwrap(),
                            _ => unreachable!(),
                        };
                        backend.synchronize().unwrap();
                        black_box(result);
                    });
                });
            }
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_add,
    bench_relu,
    bench_softmax,
    bench_summary
);

criterion_main!(benches);
