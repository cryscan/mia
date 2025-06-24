use proc_macro2::TokenStream;
use quote::quote;
use syn::{ItemFn, Stmt, parse_quote};

pub fn shader(mut input: ItemFn) -> TokenStream {
    input.sig.output = parse_quote! { -> String };

    let mut stmts: Vec<Stmt> = vec![];
    stmts.push(parse_quote! {
        let mut code = String::new();
    });
    stmts.push(parse_quote! {
        return code;
    });

    input.block.stmts = stmts;
    quote! { #input }
}
