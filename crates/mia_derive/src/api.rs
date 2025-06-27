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
        .map(|ti| quote! { #ti.ir(Access::ReadOnly) })
        .collect::<Vec<_>>();

    let generics = quote! {
        D: Device + Clone,
        U: Scalar,
        Op: TensorOp,
        F: FnOnce(InnerOp<#n, 1>) -> Op,
    };

    let fn_name = syn::Ident::new(&format!("build_api_{n}"), proc_macro2::Span::call_site());

    quote! {
        #[allow(unused)]
        pub fn #fn_name<D, U, Op, F>(
            f: F,
            mut output: Tensor<D, U>,
            #(#tensor_params: Tensor<D, impl Scalar>),*
        ) -> Tensor<D, U>
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
            let outputs = [output.ir(Access::WriteOnly)];
            let op = f(InnerOp::new(inputs, outputs));
            ops.push(Box::new(op));
            output.replace_ops(ops);

            output
        }
    }
}
