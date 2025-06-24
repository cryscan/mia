use proc_macro::TokenStream;
use syn::{DeriveInput, ItemFn, LitInt, parse_macro_input};

mod api;
mod gpu;
mod ops;

#[proc_macro_derive(TensorOp, attributes(tensor_op))]
pub fn derive_tensor_op(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let expanded = ops::derive_tensor_op(input);
    expanded.into()
}

#[proc_macro]
pub fn build_api(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as LitInt);
    let expanded = api::build_api(input);
    expanded.into()
}

/// Transforms annotated Rust functions into shader code generators.
///
/// # Overview
/// The `shader` macro transforms annotated Rust functions into shader code generators.
/// It processes the function body and converts it into a string representation that can
/// be used as shader code.
///
/// # Features
/// - Loop unrolling with `#[unroll]` attribute
/// - Statement-by-statement code generation
/// - Support for range-based iterations
///
/// # Example
/// ```rust
/// #[shader]
/// pub fn kernel() {
///     let mut x: u32 = 0u;
///     #[unroll]
///     for i in 0..10 {
///         x = x + i;
///     }
/// }
/// ```
///
/// This will be expanded to:
/// ```rust
/// pub fn kernel() -> String {
///     let mut code = String::new();
///     writeln!(code, "var x: u32 = 0u;").unwrap();
///     for i in 0..10 {
///         writeln!(code, "x = x + {i};").unwrap();
///     }
///     code
/// }
/// ```
#[proc_macro_attribute]
pub fn shader(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let expanded = gpu::shader(input);
    expanded.into()
}
