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

    let const_fn = quote! {
        const fn round_to(offset: usize, align: usize) -> usize { offset.div_ceil(align) * align }
        const fn max(a: usize, b: usize) -> usize { [a, b][(a < b) as usize] }
    };

    let offsets = fields.iter().scan(vec![], |sizes, field| {
        let ty = &field.ty;
        let size = quote!(<#ty as #base_path::ShaderType>::SIZE);
        let align = quote!(<#ty as #base_path::ShaderType>::ALIGN);
        sizes.push(quote!(round_to(#size, #align)));

        let offset = sizes
            .iter()
            .fold(quote!(0usize), |acc, x| quote!(#acc + #x));
        Some(quote! {
            let offset = #offset;
            let align = #align;
            round_to(offset, align)
        })
    });
    let offsets = [quote!(0usize)].into_iter().chain(offsets);

    let align = fields.iter().fold(quote!(1usize), |acc, field| {
        let ty = &field.ty;
        quote!(max(#acc, <#ty as #base_path::ShaderType>::ALIGN))
    });

    let size = offsets
        .clone()
        .last()
        .map(|size| {
            quote! {
                #const_fn
                let size = { #size };
                let align = <Self as #base_path::ShaderType>::ALIGN;
                round_to(size, align)
            }
        })
        .expect("must have size");
    let align = quote! {
        #const_fn
        #align
    };
    let offsets = offsets.map(|offset| {
        quote! {
            #const_fn
            #offset
        }
    });

    let members: TokenStream = fields
        .iter()
        .zip(offsets.clone())
        .map(|(field, offset)| {
            let name = match &field.ident {
                Some(ident) => quote!(Some(stringify!(#ident).into())),
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

    let fields: TokenStream = fields
        .iter()
        .map(|field| match &field.ident {
            Some(name) => quote!(Some(stringify!(#name))),
            None => quote!(None),
        })
        .zip(offsets)
        .enumerate()
        .map(|(index, (name, offset))| {
            quote! {
                #base_path::ShaderField {
                    name: #name,
                    index: #index,
                    offset: { #offset },
                },
            }
        })
        .collect();

    quote! {
        impl #impl_generics #base_path::ShaderType for #struct_name #ty_generics #where_clause {
            const SIZE: usize = { #size };
            const ALIGN: usize = { #align };

            fn shader_type(types: &mut ::naga::UniqueArena<::naga::Type>) -> ::naga::Handle<::naga::Type> {
                let name = Some(stringify!(#struct_name).into());
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
        }

        impl #impl_generics #base_path::ShaderStruct for #struct_name #ty_generics #where_clause {
            const FIELDS: &'static [#base_path::ShaderField] = &[
                #fields
            ];
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
