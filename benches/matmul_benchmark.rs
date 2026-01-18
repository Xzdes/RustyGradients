use rusty_gradients::backend::{Backend, Device};
use rusty_gradients::backend::cpu::CpuBackend;
use rusty_gradients::tensor_v2::TensorV2;
use std::time::Instant;

fn benchmark_matmul(size: usize, name: &str) {
    let device = Device::cpu();

    // Create random tensors
    let a = TensorV2::randn(&[size, size], false);
    let b = TensorV2::randn(&[size, size], false);

    // Warm-up
    for _ in 0..3 {
        let _ = a.matmul(&b).unwrap();
    }

    // Benchmark
    let iterations = 10;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = a.matmul(&b).unwrap();
    }

    let duration = start.elapsed();
    let avg_ms = duration.as_millis() as f64 / iterations as f64;

    println!("{}: {}x{} matmul: {:.2} ms/iter", name, size, size, avg_ms);
}

fn benchmark_batched_matmul(batch_size: usize, size: usize) {
    let backend = CpuBackend::new();

    // Create batched tensors [batch, size, size]
    let a = TensorV2::randn(&[batch_size, size, size], false);
    let b = TensorV2::randn(&[batch_size, size, size], false);

    // Get CPU data for backend call
    let a_data = a.to_cpu_data().unwrap();
    let b_data = b.to_cpu_data().unwrap();

    // Warm-up
    for _ in 0..3 {
        let _ = backend.matmul(&a_data, &b_data).unwrap();
    }

    // Benchmark
    let iterations = 10;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = backend.matmul(&a_data, &b_data).unwrap();
    }

    let duration = start.elapsed();
    let avg_ms = duration.as_millis() as f64 / iterations as f64;

    #[cfg(feature = "cpu")]
    let parallel_info = "(parallel)";
    #[cfg(not(feature = "cpu"))]
    let parallel_info = "(sequential)";

    println!("Batched {}: [{}x{}x{}] matmul: {:.2} ms/iter",
             parallel_info, batch_size, size, size, avg_ms);
}

fn main() {
    println!("=== RustyGradients Backend Performance Benchmark ===\n");

    println!("Matrix Multiplication (2D):");
    benchmark_matmul(64, "CPU");
    benchmark_matmul(128, "CPU");
    benchmark_matmul(256, "CPU");
    benchmark_matmul(512, "CPU");

    println!("\nBatched Matrix Multiplication (3D):");
    benchmark_batched_matmul(8, 64);
    benchmark_batched_matmul(16, 128);
    benchmark_batched_matmul(32, 64);

    println!("\nMulti-Head Attention Simulation (4D):");
    let backend = CpuBackend::new();
    let batch = 4;
    let heads = 8;
    let seq_len = 64;
    let head_dim = 64;

    let q = TensorV2::randn(&[batch, heads, seq_len, head_dim], false);
    let k = TensorV2::randn(&[batch, heads, head_dim, seq_len], false);

    let q_data = q.to_cpu_data().unwrap();
    let k_data = k.to_cpu_data().unwrap();

    // Warm-up
    for _ in 0..3 {
        let _ = backend.matmul(&q_data, &k_data).unwrap();
    }

    let iterations = 10;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = backend.matmul(&q_data, &k_data).unwrap();
    }

    let duration = start.elapsed();
    let avg_ms = duration.as_millis() as f64 / iterations as f64;

    #[cfg(feature = "cpu")]
    let parallel_info = "(rayon parallel)";
    #[cfg(not(feature = "cpu"))]
    let parallel_info = "(sequential)";

    println!("Attention QK^T {}: [{}x{}x{}x{}] @ [{}x{}x{}x{}]: {:.2} ms/iter",
             parallel_info,
             batch, heads, seq_len, head_dim,
             batch, heads, head_dim, seq_len,
             avg_ms);

    println!("\n=== Benchmark Complete ===");

    #[cfg(feature = "cpu")]
    println!("\nNote: Rayon parallelization ENABLED (feature 'cpu')");
    #[cfg(not(feature = "cpu"))]
    println!("\nNote: Rayon parallelization DISABLED. Run with --features cpu for speedup");
}
