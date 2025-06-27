use proc_macro::TokenStream;
use syn::{DeriveInput, LitInt, parse_macro_input};

mod api;
mod ops;
mod shader;

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

#[proc_macro_derive(ShaderType, attributes(shader))]
pub fn shader_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let expanded = shader::derive_shader_type(input);
    expanded.into()
}
