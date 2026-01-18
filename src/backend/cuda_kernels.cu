///! Custom CUDA Kernels for RustyGradients
///!
///! High-performance CUDA kernels for deep learning operations.
///!
///! Expected Performance:
///! - Elementwise ops: 20-50x vs CPU
///! - Fused operations: 5-10x vs unfused
///!
///! Compile with:
///! nvcc -ptx -O3 cuda_kernels.cu -o cuda_kernels.ptx

#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// Elementwise Operations (20-50x speedup expected)
// ============================================================================

/// Elementwise addition: c[i] = a[i] + b[i]
/// Expected: 20-30x vs CPU
extern "C" __global__ void elementwise_add(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/// Elementwise multiplication: c[i] = a[i] * b[i]
/// Expected: 20-30x vs CPU
extern "C" __global__ void elementwise_mul(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

/// Elementwise subtraction: c[i] = a[i] - b[i]
/// Expected: 20-30x vs CPU
extern "C" __global__ void elementwise_sub(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

/// Elementwise division: c[i] = a[i] / b[i]
/// Expected: 20-30x vs CPU
extern "C" __global__ void elementwise_div(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

// ============================================================================
// Activation Functions (30-50x speedup expected)
// ============================================================================

/// ReLU activation: y[i] = max(0, x[i])
/// Expected: 30-50x vs CPU
extern "C" __global__ void relu_forward(
    const float* x,
    float* y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}

/// ReLU backward: grad_x[i] = grad_y[i] if x[i] > 0 else 0
extern "C" __global__ void relu_backward(
    const float* grad_y,
    const float* x,
    float* grad_x,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_x[idx] = (x[idx] > 0.0f) ? grad_y[idx] : 0.0f;
    }
}

/// Sigmoid activation: y[i] = 1 / (1 + exp(-x[i]))
/// Expected: 20-40x vs CPU
extern "C" __global__ void sigmoid_forward(
    const float* x,
    float* y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

/// GELU activation (approximate): y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// Expected: 20-30x vs CPU
extern "C" __global__ void gelu_forward(
    const float* x,
    float* y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));
        y[idx] = val * cdf;
    }
}

/// Exponential: y[i] = exp(x[i])
/// Expected: 20-30x vs CPU
extern "C" __global__ void exp_forward(
    const float* x,
    float* y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = expf(x[idx]);
    }
}

/// Natural logarithm: y[i] = log(x[i])
/// Expected: 20-30x vs CPU
extern "C" __global__ void log_forward(
    const float* x,
    float* y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = logf(x[idx]);
    }
}

/// Power: y[i] = x[i] ^ power
/// Expected: 20-30x vs CPU
extern "C" __global__ void powf_forward(
    const float* x,
    float* y,
    float power,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = powf(x[idx], power);
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

/// Sum reduction along last axis (2D tensor)
/// Input: [batch, n], Output: [batch]
/// Uses shared memory for efficient reduction
extern "C" __global__ void sum_reduce(
    const float* input,
    float* output,
    int batch,
    int n
) {
    extern __shared__ float sdata[];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch) return;

    // Load data into shared memory
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += input[batch_idx * n + i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        output[batch_idx] = sdata[0];
    }
}

// ============================================================================
// Softmax (Fused Kernel - 10-20x speedup)
// ============================================================================

/// Softmax along last axis (numerically stable)
/// Input: [batch, n], Output: [batch, n]
/// Fuses: max reduction + exp + sum + divide
/// Expected: 10-20x vs unfused CPU implementation
extern "C" __global__ void softmax_forward(
    const float* input,
    float* output,
    int batch,
    int n
) {
    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_val = &shared[1];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch) return;

    const float* x = &input[batch_idx * n];
    float* y = &output[batch_idx * n];

    // Step 1: Find max (for numerical stability)
    float local_max = -INFINITY;
    for (int i = tid; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }

    // Reduce max across block
    __shared__ float max_shared[256];
    max_shared[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            max_shared[tid] = fmaxf(max_shared[tid], max_shared[tid + s]);
        }
        __syncthreads();
    }

    float max_value = max_shared[0];
    __syncthreads();

    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = expf(x[i] - max_value);
        y[i] = val;  // Store exp values
        local_sum += val;
    }

    // Reduce sum across block
    __shared__ float sum_shared[256];
    sum_shared[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_shared[tid] += sum_shared[tid + s];
        }
        __syncthreads();
    }

    float sum_value = sum_shared[0];
    __syncthreads();

    // Step 3: Normalize
    for (int i = tid; i < n; i += blockDim.x) {
        y[i] /= sum_value;
    }
}

// ============================================================================
// LayerNorm (Fused Kernel - 5-10x speedup)
// ============================================================================

/// Layer Normalization (fused)
/// y = gamma * (x - mean) / sqrt(var + eps) + beta
/// Fuses: mean + variance + normalize + scale + shift
/// Expected: 5-10x vs unfused CPU implementation
extern "C" __global__ void layernorm_forward(
    const float* input,      // [batch, n]
    const float* gamma,      // [n]
    const float* beta,       // [n]
    float* output,           // [batch, n]
    int batch,
    int n,
    float epsilon
) {
    extern __shared__ float shared[];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch) return;

    const float* x = &input[batch_idx * n];
    float* y = &output[batch_idx * n];

    // Step 1: Compute mean
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local_sum += x[i];
    }

    __shared__ float mean_shared[256];
    mean_shared[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            mean_shared[tid] += mean_shared[tid + s];
        }
        __syncthreads();
    }

    float mean = mean_shared[0] / n;
    __syncthreads();

    // Step 2: Compute variance
    float local_var = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = x[i] - mean;
        local_var += diff * diff;
    }

    __shared__ float var_shared[256];
    var_shared[tid] = local_var;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            var_shared[tid] += var_shared[tid + s];
        }
        __syncthreads();
    }

    float variance = var_shared[0] / n;
    float std = sqrtf(variance + epsilon);
    __syncthreads();

    // Step 3: Normalize + scale + shift
    for (int i = tid; i < n; i += blockDim.x) {
        float normalized = (x[i] - mean) / std;
        y[i] = gamma[i] * normalized + beta[i];
    }
}

// ============================================================================
// Attention Helpers (for FlashAttention - Week 4)
// ============================================================================

/// Scaled dot-product attention (basic version)
/// Q: [batch, heads, seq, head_dim]
/// K: [batch, heads, seq, head_dim]
/// V: [batch, heads, seq, head_dim]
/// This is a placeholder - full FlashAttention will be implemented in Week 4
extern "C" __global__ void attention_qk_matmul(
    const float* Q,
    const float* K,
    float* scores,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // Placeholder for FlashAttention
    // Full implementation in Week 4
}

// ============================================================================
// Utility Kernels
// ============================================================================

/// Fill tensor with constant value
extern "C" __global__ void fill_constant(
    float* data,
    float value,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

/// Copy tensor
extern "C" __global__ void copy_tensor(
    const float* src,
    float* dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

/// Scalar multiply: y[i] = alpha * x[i]
extern "C" __global__ void scalar_mul(
    const float* x,
    float* y,
    float alpha,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx];
    }
}

/// Scalar add: y[i] = x[i] + alpha
extern "C" __global__ void scalar_add(
    const float* x,
    float* y,
    float alpha,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] + alpha;
    }
}
