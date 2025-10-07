use proc_macro2::TokenStream;
use quote::quote;
use syn::LitInt;

pub fn build_api(input: LitInt) -> TokenStream {
    let n = input.base10_parse::<usize>().unwrap();

    let tensor_params = (0..n)
        .map(|i| syn::Ident::new(&format!("t{i}"), proc_macro2::Span::call_site()))
        .collect::<Vec<_>>();

    let tape_clones = tensor_params
        .iter()
        .map(|ti| quote! { #ti.tape().ops.clone() })
        .collect::<Vec<_>>();

    let ir_calls = tensor_params
        .iter()
        .map(|ti| quote! { #ti.ir(crate::loom::ops::Access::ReadOnly) })
        .collect::<Vec<_>>();

    let generics = quote! {
        D: crate::loom::device::Device + Clone,
        U: crate::loom::num::Scalar,
        Op: crate::loom::ops::TensorOp,
        F: FnOnce(crate::loom::ops::InnerOp<#n, 1>) -> Op,
    };

    let fn_name = syn::Ident::new(&format!("build_api_{n}"), proc_macro2::Span::call_site());

    quote! {
        #[allow(unused)]
        pub fn #fn_name<D, U, Op, F>(
            f: F,
            mut output: crate::loom::tensor::Tensor<D, U>,
            #(#tensor_params: crate::loom::tensor::Tensor<D, impl crate::loom::num::Scalar>),*
        ) -> crate::loom::tensor::Tensor<D, U>
        where
            #generics
        {
            use ::itertools::Itertools;

            let mut ops = vec![#(#tape_clones),*]
                .concat()
                .into_iter()
                .unique()
                .collect_vec();

            let inputs = [#(#ir_calls),*];
            let outputs = [output.ir(crate::loom::ops::Access::WriteOnly)];
            let op = f(crate::loom::ops::InnerOp::new(inputs, outputs));
            ops.push(::std::boxed::Box::new(op));
            output.replace_ops(ops);

            output
        }
    }
}
