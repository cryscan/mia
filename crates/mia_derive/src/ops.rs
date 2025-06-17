use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    DeriveInput, Fields, LitStr, Path, Token, WherePredicate, punctuated::Punctuated,
    spanned::Spanned,
};

pub fn derive_tensor_op(input: DeriveInput) -> TokenStream {
    // retrieve struct field information
    let fields = match &input.data {
        syn::Data::Struct(data_struct) => &data_struct.fields,
        _ => {
            return syn::Error::new(input.span(), "`TensorOp` can only be derived for structs")
                .to_compile_error();
        }
    };

    // determine the field access expression for trait forwarding
    let forward_access = match fields {
        // tuple struct: must have exactly one field
        Fields::Unnamed(fields_unnamed) => {
            if fields_unnamed.unnamed.len() != 1 {
                return syn::Error::new(
                    fields_unnamed.span(),
                    "tuple structs must have exactly one field",
                )
                .to_compile_error();
            }
            // access the single tuple field
            quote! { self.0 }
        }
        // named struct: require exactly one #[tensor_op] attribute
        Fields::Named(fields_named) => {
            // collect fields with #[tensor_op] attribute
            let marked_fields: Vec<_> = fields_named
                .named
                .iter()
                .filter(|f| f.attrs.iter().any(|a| a.path().is_ident("tensor_op")))
                .collect();

            // ensure exactly one marked field exists
            if marked_fields.len() != 1 {
                let msg = match marked_fields.len() {
                    0 => "no field marked with #[tensor_op] attribute",
                    _ => "multiple fields marked with #[tensor_op] attribute",
                };
                return syn::Error::new(fields_named.span(), msg).to_compile_error();
            }

            // identifier of the marked field
            let field_ident = &marked_fields[0].ident;
            quote! { self.#field_ident }
        }
        // unit structs are not supported
        Fields::Unit => {
            return syn::Error::new(
                input.span(),
                "unit structs are not supported by `TensorOp` derive",
            )
            .to_compile_error();
        }
    };

    let name = input.ident;

    // parse tensor_op attributes
    let mut crate_name = None;
    let mut user_bounds = Punctuated::<WherePredicate, Token![,]>::new();
    for attr in &input.attrs {
        if !attr.path().is_ident("tensor_op") {
            continue;
        }

        let result = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("crate") {
                let value = meta.value()?;
                let s: LitStr = value.parse()?;
                crate_name = Some(s.parse::<Path>()?);
                Ok(())
            } else if meta.path.is_ident("bound") {
                let value = meta.value()?;
                let s: LitStr = value.parse()?;
                let predicates =
                    s.parse_with(Punctuated::<WherePredicate, Token![,]>::parse_terminated)?;
                user_bounds.extend(predicates);
                Ok(())
            } else {
                Err(meta.error("unexpected attribute; supported are `crate` and `bound`"))
            }
        });

        if let Err(err) = result {
            return err.to_compile_error();
        }
    }
    // determine the base path for trait implementation
    let base_path = match crate_name {
        Some(path) => quote!(#path::loom::ops),
        None => quote!(::mia::loom::ops),
    };

    // handle struct generics
    let generics = &input.generics;
    let (impl_generics, ty_generics, _) = generics.split_for_impl();

    // prepare where clause
    let mut generics = generics.clone();
    let where_clause = generics.make_where_clause();
    where_clause.predicates.extend(user_bounds);

    quote! {
        impl #impl_generics #base_path::TensorOp for #name #ty_generics #where_clause {
            fn id(&self) -> #base_path::TensorOpId {
                #forward_access.id()
            }

            fn io(&self) -> Vec<#base_path::TensorIr> {
                #forward_access.io()
            }
        }
    }
}
