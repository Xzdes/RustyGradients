///! CUDA Backend Demonstration
///!
///! This example demonstrates basic CUDA operations:
///! - Device initialization
///! - Memory transfer (CPU ↔ GPU)
///! - Matrix multiplication with cuBLAS
///! - Performance comparison
///!
///! Run with:
///! ```bash
///! cargo run --example cuda_demo --features cuda --release
///! ```

use rusty_gradients::backend::{Backend, Device};

#[cfg(feature = "cuda")]
use rusty_gradients::backend::cuda::CudaBackend;

fn main() {
    println!("========================================");
    println!("🚀 RustyGradients CUDA Backend Demo");
    println!("========================================\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("❌ CUDA feature not enabled!");
        println!("\nTo run this demo, compile with:");
        println!("   cargo run --example cuda_demo --features cuda --release");
        return;
    }

    #[cfg(feature = "cuda")]
    {
        demo_cuda();
    }
}

#[cfg(feature = "cuda")]
fn demo_cuda() {
    // Check CUDA availability
    println!("📊 Checking CUDA availability...");
    let device_count = CudaBackend::device_count();
    println!("   Found {} CUDA device(s)", device_count);

    if device_count == 0 {
        println!("\n❌ No CUDA devices found!");
        println!("   Make sure you have:");
        println!("   1. NVIDIA GPU installed");
        println!("   2. CUDA drivers installed");
        println!("   3. Proper CUDA toolkit (12.0+)");
        return;
    }

    println!("\n✅ CUDA is available!\n");

    // Initialize CUDA backend
    println!("🔧 Initializing CUDA backend on GPU 0...");
    let backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            println!("❌ Failed to initialize CUDA: {:?}", e);
            return;
        }
    };
    println!("   ✅ CUDA backend ready!\n");

    // Test 1: Memory Operations
    println!("📦 Test 1: Memory Operations");
    println!("   Creating 1000x1000 matrix of zeros on GPU...");

    let zeros = backend.zeros(&[1000, 1000]).expect("Failed to create zeros");
    println!("   ✅ Created successfully");

    let data = backend.to_vec(&zeros).expect("Failed to copy back");
    let sum: f32 = data.iter().sum();
    println!("   Verification: sum = {} (should be 0.0)", sum);

    assert!((sum - 0.0).abs() < 1e-5, "Zeros test failed!");
    println!("   ✅ Memory operations work!\n");

    // Test 2: Small Matrix Multiplication
    println!("📐 Test 2: Matrix Multiplication (2x2)");
    println!("   A = [[1, 2], [3, 4]]");
    println!("   B = [[5, 6], [7, 8]]");

    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![5.0, 6.0, 7.0, 8.0];

    let a = backend.from_slice(&a_data, &[2, 2]).expect("Failed to create A");
    let b = backend.from_slice(&b_data, &[2, 2]).expect("Failed to create B");

    println!("   Computing C = A @ B on GPU...");
    let c = backend.matmul(&a, &b).expect("Matmul failed");
    backend.synchronize().expect("Sync failed");

    let result = backend.to_vec(&c).expect("Failed to copy result");

    println!("   Result:");
    println!("   C = [[{:.1}, {:.1}], [{:.1}, {:.1}]]",
        result[0], result[1], result[2], result[3]);

    // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    println!("   Expected: [[19.0, 22.0], [43.0, 50.0]]");

    assert!((result[0] - 19.0).abs() < 1e-5, "Result mismatch at [0,0]");
    assert!((result[1] - 22.0).abs() < 1e-5, "Result mismatch at [0,1]");
    assert!((result[2] - 43.0).abs() < 1e-5, "Result mismatch at [1,0]");
    assert!((result[3] - 50.0).abs() < 1e-5, "Result mismatch at [1,1]");

    println!("   ✅ Matrix multiplication works!\n");

    // Test 3: Performance Benchmark
    println!("⚡ Test 3: Performance Benchmark (1024x1024)");

    let size = 1024;
    println!("   Creating {}x{} random matrices...", size, size);

    let a_large: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.001).collect();
    let b_large: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.002).collect();

    let a_gpu = backend.from_slice(&a_large, &[size, size]).expect("Failed to create A");
    let b_gpu = backend.from_slice(&b_large, &[size, size]).expect("Failed to create B");

    println!("   Warming up GPU...");
    for _ in 0..3 {
        let _ = backend.matmul(&a_gpu, &b_gpu).expect("Warmup matmul failed");
        backend.synchronize().expect("Sync failed");
    }

    println!("   Running benchmark (10 iterations)...");
    let iterations = 10;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        let _ = backend.matmul(&a_gpu, &b_gpu).expect("Benchmark matmul failed");
        backend.synchronize().expect("Sync failed");
    }

    let elapsed = start.elapsed();
    let avg_time = elapsed.as_secs_f64() / iterations as f64;

    // Calculate GFLOPS
    // For matmul: operations = 2*m*k*n (multiply + add)
    let ops = 2.0 * size as f64 * size as f64 * size as f64;
    let gflops = ops / avg_time / 1e9;

    println!("\n   Results:");
    println!("   Average time: {:.3} ms", avg_time * 1000.0);
    println!("   Throughput: {:.1} GFLOPS", gflops);

    if gflops > 1000.0 {
        println!("   🏆 Excellent performance! (>1 TFLOPS)");
    } else if gflops > 500.0 {
        println!("   ✅ Good performance (>500 GFLOPS)");
    } else {
        println!("   ⚠️  Lower than expected (<500 GFLOPS)");
        println!("      This might be normal for older GPUs or CPU fallback");
    }

    // Comparison with expected CPU performance
    let cpu_gflops = 77.0; // Naive CPU performance
    let speedup = gflops / cpu_gflops;
    println!("\n   Speedup vs naive CPU: {:.1}x", speedup);

    if speedup > 50.0 {
        println!("   🚀 Amazing speedup! (>50x)");
    } else if speedup > 10.0 {
        println!("   ✅ Great speedup! (>10x)");
    } else if speedup > 5.0 {
        println!("   📈 Good speedup (>5x)");
    } else {
        println!("   ⚠️  Lower speedup than expected");
    }

    println!("\n========================================");
    println!("✅ All CUDA tests passed!");
    println!("========================================\n");

    println!("💡 Next steps:");
    println!("   1. Run full benchmark: cargo bench --bench cuda_comparison --features cuda");
    println!("   2. Try training GPT with CUDA backend");
    println!("   3. Compare with CPU BLAS performance\n");
}
