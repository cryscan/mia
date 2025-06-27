use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    Attribute, DeriveInput, LitStr, Path, Token, WherePredicate, punctuated::Punctuated,
    spanned::Spanned,
};

pub fn derive_shader_type(input: DeriveInput) -> TokenStream {
    // parse shader_type attributes
    let mut crate_name = None;
    let mut user_bounds = Punctuated::<WherePredicate, Token![,]>::new();
    for attr in &input.attrs {
        if !attr.path().is_ident("shader") {
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
        Some(path) => quote!(#path::hal::gpu::shader),
        None => quote!(::mia::hal::gpu::shader),
    };

    // handle struct generics
    let generics = &input.generics;
    let (impl_generics, ty_generics, _) = generics.split_for_impl();

    // prepare where clause
    let mut generics = generics.clone();
    let where_clause = generics.make_where_clause();
    where_clause.predicates.extend(user_bounds);

    let struct_name = &input.ident;

    // retrieve struct field information
    let fields = match &input.data {
        syn::Data::Struct(data_struct) => &data_struct.fields,
        _ => {
            return syn::Error::new(input.span(), "`ShaderType` can only be derived for structs")
                .to_compile_error();
        }
    };

    let offsets = fields.iter().scan(quote!(0usize), |offset, field| {
        let ty = &field.ty;
        let size = quote!(<#ty as #base_path::ShaderType>::SIZE);
        let align = quote!(<#ty as #base_path::ShaderType>::ALIGN);
        let output = quote! {
            let offset = { #offset };
            let size = #size;
            let align = #align;
            offset.div_ceil(align) * align
        };
        let next = quote! {
            let offset = { #output };
            let size = #size;
            offset + size
        };
        *offset = next.clone();
        Some((output, next))
    });

    let size = offsets
        .clone()
        .map(|(_, x)| x)
        .last()
        .expect("must have size");
    let align = fields.iter().fold(quote!(1usize), |acc, field| {
        let ty = &field.ty;
        quote!(max(#acc, <#ty as #base_path::ShaderType>::ALIGN))
    });

    // define the max function
    let align = quote! {
        const fn max(a: usize, b: usize) -> usize { [a, b][(a < b) as usize] }
        #align
    };
    // round up size to multiples of align
    let size = quote! {
        let size = { #size };
        let align = { #align };
        size.div_ceil(align) * align
    };

    let members: TokenStream = fields
        .iter()
        .zip(offsets.clone())
        .map(|(field, (offset, _))| {
            let field_name = &field.ident;
            let name = match field_name {
                Some(ident) => {
                    let name = ident.to_string();
                    quote!(Some(#name.to_string()))
                }
                None => quote!(None),
            };
            let ty = &field.ty;
            let ty = quote!(<#ty as #base_path::ShaderType>::shader_type(types));
            let binding = parse_binding(&field.attrs);
            // let offset = quote!(::bytemuck::offset_of!(#struct_name, #field_name) as u32);
            quote! {
                members.push(::naga::StructMember {
                    name: #name,
                    ty: #ty,
                    binding: #binding,
                    offset: { #offset } as u32,
                });
            }
        })
        .collect();

    let field_offsets: TokenStream = fields
        .iter()
        .zip(offsets.clone())
        .enumerate()
        .map(|(index, (field, (offset, _)))| {
            let name = field
                .ident
                .clone()
                .map(|ident| ident.to_string())
                .unwrap_or_else(|| format!(".{index}"));
            quote! {
                #name => { #offset },
            }
        })
        .collect();

    let name = struct_name.to_string();
    quote! {
        impl #impl_generics #base_path::ShaderType for #struct_name #ty_generics #where_clause {
            const SIZE: usize = { #size };
            const ALIGN: usize = { #align };

            fn shader_type(types: &mut ::naga::UniqueArena<::naga::Type>) -> ::naga::Handle<::naga::Type> {
                let name = Some(#name.to_string());
                let members = {
                    let mut members = Vec::new();
                    #members;
                    members
                };
                let span = { #size } as u32;
                let inner = ::naga::TypeInner::Struct { members, span };
                let r#type = ::naga::Type { name, inner };
                types.insert(r#type, Default::default())
            }

            fn shader_field_offset(field: impl AsRef<str>) -> usize {
                let field = field.as_ref();
                match field {
                    #field_offsets
                    _ => panic!("unknown field"),
                }
            }
        }
    }
}

fn parse_binding(attrs: &[Attribute]) -> TokenStream {
    let mut builtin = None;

    for attr in attrs.iter() {
        if !attr.path().is_ident("shader") {
            continue;
        }

        let result = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("builtin") {
                let value = meta.value()?;
                let s: LitStr = value.parse()?;
                builtin = match s.value().as_str() {
                    "uid" => Some(quote!(::naga::BuiltIn::GlobalInvocationId)),
                    "tid" => Some(quote!(::naga::BuiltIn::LocalInvocationId)),
                    "index" => Some(quote!(::naga::BuiltIn::LocalInvocationIndex)),
                    "bid" => Some(quote!(::naga::BuiltIn::WorkGroupId)),
                    "bs" => Some(quote!(::naga::BuiltIn::WorkGroupSize)),
                    "nb" => Some(quote!(::naga::BuiltIn::NumWorkGroups)),
                    _ => return Err(meta.error("unexpected builtin")),
                };
                Ok(())
            } else {
                Err(meta.error("unexpected attribute; supported is `builtin`"))
            }
        });

        if let Err(err) = result {
            return err.to_compile_error();
        }
    }

    match builtin {
        Some(builtin) => quote!(Some(::naga::Binding::BuiltIn(#builtin))),
        None => quote!(None),
    }
}
