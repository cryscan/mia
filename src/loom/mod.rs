//! The `loom` module provides foundational abstractions for tensor computation and neural network operations.
//! It offers a unified interface for hardware-accelerated operations across CPU and GPU devices, with strong support for asynchronous execution and memory management.
//!
//! ## Key Components
//! 1. **Tensor Operations**:
//!    - Defines tensor layouts (`Layout`) and algebraic operations.
//!    - Supports coalescing, composition, and tiling operations.
//!    - Enables shape/strides manipulation for multidimensional data.
//!
//! 2. **Device Abstraction**:
//!    - Hardware-agnostic execution via `Device` trait.
//!    - Logical tensor allocator with automatic reuse/disposal logic.
//!
//! 3. **Numerical System**:
//!    - Scalar types (`f32`, `f16`, `u8`, etc.) and packed types (`F32x4`, `U4x8`).
//!    - Data type metadata (`DataType`) for tensor element representation.
//!
//! 4. **Execution Model**:
//!    - Asynchronous operation execution.
//!    - Operation tapes (`TensorTape`) for dependency tracking and deferred execution.
//!    - Hardware-specific backends.
//!
//! ## Design Principles
//! - **Portability**: WASM support via `wasm_bindgen_futures`.
//! - **Efficiency**: Optimized memory reuse through `Allocator`.
//! - **Extensibility**: Custom ops via `TensorOp` trait.
//! - **Safety**: Bounds checking and access validation.
//!
//! This module serves as the computational backbone for higher-level neural network constructs.

#![cfg_attr(target_arch = "wasm32", allow(async_fn_in_trait))]

pub mod device;
pub mod layout;
pub mod num;
pub mod ops;
pub mod platform;
pub mod slice;
pub mod tensor;
