///! Демонстрация TensorV2 с multi-backend поддержкой

use rusty_gradients::backend::Device;
use rusty_gradients::tensor_v2::TensorV2;

fn main() {
    println!("=== RustyGradients TensorV2 Demo ===\n");

    // 1. Создание тензоров на разных устройствах
    println!("1. Creating tensors on different devices:");

    let cpu_device = Device::cpu();
    println!("   CPU device: {:?}", cpu_device.device_type());

    let t1 = TensorV2::zeros(&[2, 3], true, cpu_device.clone()).unwrap();
    println!("   t1 (zeros): {:?}", t1);

    let t2 = TensorV2::ones(&[2, 3], false, cpu_device.clone()).unwrap();
    println!("   t2 (ones): {:?}", t2);

    let t3 = TensorV2::randn(&[3, 4], false);
    println!("   t3 (randn): {:?}", t3);

    // 2. Базовые операции
    println!("\n2. Basic operations:");

    let a = TensorV2::ones(&[2, 2], false, Device::cpu()).unwrap();
    let b = TensorV2::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false).unwrap();

    println!("   a = {:?}", a.shape());
    println!("   b = {:?}", b.shape());

    let c = a.add(&b).unwrap();
    println!("   a + b = tensor with shape {:?}", c.shape());

    let d = a.mul(&b).unwrap();
    println!("   a * b = tensor with shape {:?}", d.shape());

    // 3. Matrix multiplication
    println!("\n3. Matrix multiplication:");

    let x = TensorV2::randn(&[3, 4], false);
    let w = TensorV2::randn(&[4, 5], false);

    println!("   x shape: {:?}", x.shape());
    println!("   w shape: {:?}", w.shape());

    let y = x.matmul(&w).unwrap();
    println!("   y = x @ w, shape: {:?}", y.shape());

    // 4. Activations
    println!("\n4. Activation functions:");

    let z = TensorV2::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5], false).unwrap();
    println!("   z = {:?}", z.shape());

    let relu_out = z.relu().unwrap();
    println!("   relu(z) shape: {:?}", relu_out.shape());

    // 5. Reshape and transpose
    println!("\n5. Shape operations:");

    let original = TensorV2::randn(&[2, 3, 4], false);
    println!("   original shape: {:?}", original.shape());

    let reshaped = original.reshape(&[2, 12]).unwrap();
    println!("   reshaped to: {:?}", reshaped.shape());

    let matrix = TensorV2::randn(&[3, 4], false);
    let transposed = matrix.transpose(0, 1).unwrap();
    println!("   matrix {:?} -> transposed {:?}", matrix.shape(), transposed.shape());

    // 6. Batched operations (for neural networks)
    println!("\n6. Batched operations (mini-batch processing):");

    let batch_size = 32;
    let input_dim = 128;
    let hidden_dim = 256;

    let x_batch = TensorV2::randn(&[batch_size, input_dim], false);
    let weights = TensorV2::randn(&[input_dim, hidden_dim], false);
    let bias = TensorV2::ones(&[hidden_dim], false, Device::cpu()).unwrap();

    println!("   Input batch: {:?}", x_batch.shape());
    println!("   Weights: {:?}", weights.shape());

    // Linear layer: y = x @ W + b
    let linear_out = x_batch.matmul(&weights).unwrap();
    println!("   After matmul: {:?}", linear_out.shape());

    let activated = linear_out.relu().unwrap();
    println!("   After ReLU: {:?}", activated.shape());

    // 7. Multi-head attention dimensions
    println!("\n7. Multi-head attention pattern:");

    let batch = 4;
    let heads = 8;
    let seq_len = 64;
    let head_dim = 64;

    let q = TensorV2::randn(&[batch, heads, seq_len, head_dim], false);
    let k = TensorV2::randn(&[batch, heads, seq_len, head_dim], false);

    println!("   Query shape: {:?}", q.shape());
    println!("   Key shape: {:?}", k.shape());

    // Transpose last two dims of k for attention scores
    let k_t = k.transpose(2, 3).unwrap();
    println!("   Key transposed: {:?}", k_t.shape());

    let scores = q.matmul(&k_t).unwrap();
    println!("   Attention scores (Q @ K^T): {:?}", scores.shape());

    let attn = scores.softmax().unwrap();
    println!("   After softmax: {:?}", attn.shape());

    println!("\n=== Demo Complete ===");
    println!("\nKey Features:");
    println!("  ✓ Device abstraction (CPU/CUDA/Metal/WASM)");
    println!("  ✓ Zero-cost backend dispatch");
    println!("  ✓ Rayon parallelization for batched operations");
    println!("  ✓ PyTorch-like API");
    println!("  ✓ Ready for gradient computation (autograd)");

    #[cfg(feature = "cpu")]
    println!("\n[Rayon parallelization: ENABLED]");
    #[cfg(not(feature = "cpu"))]
    println!("\n[Rayon parallelization: DISABLED - use --features cpu]");
}
