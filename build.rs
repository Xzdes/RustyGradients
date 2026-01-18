///! Build script for RustyGradients
///!
///! Compiles CUDA kernels to PTX when the `cuda` feature is enabled.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    #[cfg(feature = "cuda")]
    compile_cuda_kernels();
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernels() {
    println!("cargo:rerun-if-changed=src/backend/cuda_kernels.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_source = PathBuf::from("src/backend/cuda_kernels.cu");
    let ptx_output = out_dir.join("cuda_kernels.ptx");

    // Try to find nvcc
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());

    println!("cargo:warning=Compiling CUDA kernels with nvcc...");
    println!("cargo:warning=  Source: {:?}", cuda_source);
    println!("cargo:warning=  Output: {:?}", ptx_output);

    // Compile CUDA kernels to PTX
    let status = Command::new(&nvcc)
        .args(&[
            "-ptx",                          // Generate PTX intermediate
            "-O3",                            // Optimize
            "--use_fast_math",                // Fast math (slight accuracy loss for speed)
            "-arch=sm_60",                    // Minimum compute capability 6.0 (Pascal+)
            "--expt-relaxed-constexpr",       // Allow relaxed constexpr
            "-o", ptx_output.to_str().unwrap(),
            cuda_source.to_str().unwrap(),
        ])
        .status();

    match status {
        Ok(status) if status.success() => {
            println!("cargo:warning=✅ CUDA kernels compiled successfully!");
            println!("cargo:warning=   PTX file: {:?}", ptx_output);
        }
        Ok(status) => {
            println!("cargo:warning=⚠️  nvcc compilation failed with status: {:?}", status);
            println!("cargo:warning=   CUDA kernels will not be available");
            println!("cargo:warning=   Install CUDA Toolkit 12.0+ and ensure nvcc is in PATH");
        }
        Err(e) => {
            println!("cargo:warning=⚠️  Failed to run nvcc: {}", e);
            println!("cargo:warning=   CUDA kernels will not be available");
            println!("cargo:warning=   Install CUDA Toolkit 12.0+ to enable CUDA support");
        }
    }

    // Tell cargo where to find the PTX file
    println!("cargo:rustc-env=CUDA_KERNELS_PTX={}", ptx_output.display());
}
