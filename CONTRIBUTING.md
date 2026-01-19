# Contributing to RustyGradients

Thank you for your interest in contributing to RustyGradients! This document provides guidelines for contributing.

## Getting Started

### Prerequisites

- Rust 1.70+ (stable)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/Xzdes/RustyGradients.git
cd RustyGradients

# Build with default features
cargo build --features "cpu serialization"

# Run tests
cargo test --lib --features "cpu serialization"
```

## Development Workflow

### Branch Strategy

- `master` - stable release branch
- Feature branches: `feature/your-feature-name`
- Bug fixes: `fix/issue-description`

### Making Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `cargo test --lib --features "cpu serialization"`
5. Run clippy: `cargo clippy --features "cpu serialization"`
6. Format code: `cargo fmt`
7. Commit with a clear message
8. Push and create a Pull Request

### Commit Messages

Follow conventional commits:

```
feat: add new activation function
fix: correct gradient calculation in matmul
docs: update README with CUDA examples
test: add tests for tokenization module
refactor: simplify backend trait
```

## Code Style

### Rust Guidelines

- Use `cargo fmt` before committing
- Run `cargo clippy` and address warnings
- Write documentation for public APIs
- Add tests for new functionality

### Documentation

- Use `///` for public API documentation
- Use `//!` for module-level documentation
- Include examples in doc comments when helpful

## Testing

### Running Tests

```bash
# Unit tests only (fastest)
cargo test --lib --features "cpu serialization"

# All tests including integration
cargo test --features "cpu serialization"

# Specific test
cargo test test_name --features "cpu serialization"
```

### Writing Tests

- Place unit tests in the same file using `#[cfg(test)]` module
- Use descriptive test names: `test_matmul_2d_shapes`
- Test edge cases and error conditions

## Feature Flags

Available features:

| Feature | Description |
|---------|-------------|
| `cpu` | CPU backend with rayon parallelization (default) |
| `cpu-blas` | BLAS acceleration for matrix operations |
| `cuda` | NVIDIA GPU support via cuBLAS |
| `metal-backend` | Apple Silicon GPU support |
| `serialization` | Safetensors model format |
| `tokenization` | BPE and HuggingFace tokenizers |
| `huggingface` | HuggingFace Hub integration |

## Project Structure

```
src/
├── backend/       # Multi-backend abstraction (CPU, CUDA, Metal)
├── core/          # Autograd and computational graph
├── nn/            # Neural network layers
├── ops/           # Tensor operations
├── models/        # Model implementations (GPT)
├── serialization/ # Model save/load (Safetensors)
├── tokenization/  # Tokenizers (BPE, HuggingFace)
└── error.rs       # Error types
```

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add entry to CHANGELOG (if applicable)
4. Request review from maintainers

### PR Checklist

- [ ] Tests pass locally
- [ ] Code is formatted (`cargo fmt`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Documentation updated (if needed)
- [ ] Commit messages follow convention

## Areas for Contribution

### High Priority

1. **CUDA Backend** - Optimize custom kernels
2. **Metal Backend** - Apple Silicon support
3. **Model Zoo** - Add LLaMA, Mistral, BERT
4. **Documentation** - Tutorials and examples

### Good First Issues

Look for issues labeled `good first issue` on GitHub.

## Questions?

- Open an issue on GitHub
- Check existing documentation in `/docs`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
