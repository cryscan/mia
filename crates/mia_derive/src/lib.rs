use proc_macro::TokenStream;
use syn::{DeriveInput, ItemFn, LitInt, parse_macro_input};

mod api;
mod ops;
mod rumia;
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
pub fn derive_shader_type(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let expanded = shader::derive_shader_type(input);
    expanded.into()
}

#[proc_macro_attribute]
pub fn rumia(attr: TokenStream, input: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as proc_macro2::TokenStream);
    let input = parse_macro_input!(input as ItemFn);
    let expanded = rumia::rumia(attr, input);
    expanded.into()
}
