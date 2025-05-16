# Mia

*A Lightweight, Cross-Platform Inference Framework in Rust*

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/cryscan/mia)

---

`mia` is a high-performance model inference framework designed for efficient execution of machine learning models across diverse hardware platforms. Built in Rust, it emphasizes type safety, memory efficiency, and cross-platform compatibility, supporting both native applications and web browsers via WebAssembly.

---

✨ **Key Features**
- Multi-Device Support:
  - CPU: Optimized tensor operations for native performance.
  - GPU: Accelerated computations via WebGPU (supports Vulkan/Metal/DirectX12 on native platforms and WebGPU in browsers).

- Cross-Platform Execution:
  - Native platforms (Windows, macOS, Linux) and web browsers (via WebAssembly).

- Efficient Tensor System:
  - Flexible memory layouts inspired by NVIDIA’s CuTe for optimal data organization.
  - Operations: Element-wise, reductions, linear algebra, and shape manipulation.

- Type-Safe Numerics:
  - Supports `f32`, `f16`, `u8`, `u16`, `u32`, and packed formats (e.g., `PackedU4x8`).

- Async-First Design:
  - Non-blocking operations with platform-specific async runtimes.

- Memory Management:
  - Buffer caching, tensor views, and layout optimizations to minimize copies and allocations.

---

🚀 **Supported Platforms**
| Device | Backend     | Native | Web (WASM) |
| ------ | ----------- | ------ | ---------- |
| CPU    | Native Rust | ✅      | ✅          |
| GPU    | WebGPU      | ✅      | ✅          |

---

📦 **Installation**
Add `mia` to your `Cargo.toml`:
```toml
[dependencies]
mia = { git = "https://github.com/cryscan/mia", branch = "main" }
```

---

🤝 **Contributing**
We welcome contributions! Check out the [Core Concepts](https://deepwiki.io/cryscan/mia) documentation and open an issue/PR to discuss ideas.

---

License: MIT
GitHub: [cryscan/mia](https://github.com/cryscan/mia)