use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    Attribute, DeriveInput, LitStr, Path, Token, WherePredicate, punctuated::Punctuated,
    spanned::Spanned,
};

macro_rules! bail {
    ($span:expr, $message:expr) => {
        return syn::Error::new($span, $message).to_compile_error()
    };
}

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
        _ => bail!(input.span(), "`ShaderType` can only be derived for structs"),
    };

    let const_fn = quote! {
        const fn round_to(offset: usize, align: usize) -> usize { offset.div_ceil(align) * align }
        const fn max(a: usize, b: usize) -> usize { [a, b][(a < b) as usize] }
    };

    let offsets = fields
        .iter()
        .map(|field| &field.ty)
        .scan(quote!(0usize), |state, ty| {
            let size = quote!(<#ty as #base_path::ShaderType>::SIZE);
            let align = quote!(<#ty as #base_path::ShaderType>::ALIGN);
            let offset = quote!(#state);
            *state = quote!(round_to(#offset + #size, #align));
            Some(offset)
        });

    // sizes of all fields
    let sizes = fields
        .iter()
        .map(|field| &field.ty)
        .map(|ty| quote!(<#ty as #base_path::ShaderType>::SIZE));

    // total size of the struct
    let size = offsets
        .clone()
        .last()
        .zip(sizes.clone().last())
        .map(|(offset, size)| quote!(round_to(#offset + #size, <Self as #base_path::ShaderType>::ALIGN)))
        .map(|size| quote!(#const_fn #size))
        .unwrap_or(quote!(0usize));

    // total alignment of the struct
    let align = fields.iter().map(|field| &field.ty).fold(
        quote!(1usize),
        |acc, ty| quote!(max(#acc, <#ty as #base_path::ShaderType>::ALIGN)),
    );
    let align = quote!(#const_fn #align);

    // offsets of all fields
    let offsets = offsets.map(|offset| quote!(#const_fn #offset));

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
            let offset = quote!({ #offset } as u32);
            quote! {
                ::naga::StructMember {
                    name: #name,
                    ty: #ty,
                    binding: #binding,
                    offset: #offset,
                },
            }
        })
        .collect();

    let field_info: TokenStream = fields
        .iter()
        .map(|field| match &field.ident {
            Some(name) => quote!(Some(stringify!(#name))),
            None => quote!(None),
        })
        .zip(offsets)
        .zip(sizes)
        .map(|((name, offset), size)| {
            let offset = quote!({ #offset });
            let size = quote!({ #size });
            quote! {
                #base_path::ShaderField {
                    name: #name,
                    offset: #offset,
                    size: #size,
                },
            }
        })
        .collect();

    // field access code for shader_bytes implementation
    let field_bytes: TokenStream = fields
        .iter()
        .enumerate()
        .map(|(index, field)| {
            let field_name = match &field.ident {
                Some(ident) => quote!(#ident),
                None => {
                    let index = syn::Index::from(index);
                    quote!(#index)
                }
            };
            quote! {
                let field_data = self.#field_name.shader_bytes();
                let field_info = &Self::FIELDS[#index];
                let start = field_info.offset;
                let end = start + field_info.size;
                data[start..end].copy_from_slice(&field_data[..]);
            }
        })
        .collect();

    quote! {
        impl #impl_generics #base_path::ShaderType for #struct_name #ty_generics #where_clause {
            const SIZE: usize = { #size };
            const ALIGN: usize = { #align };

            fn shader_type(types: &mut ::naga::UniqueArena<::naga::Type>) -> ::naga::Handle<::naga::Type> {
                let name = Some(stringify!(#struct_name).into());
                let members = vec![#members];
                let span = { #size } as u32;
                let inner = ::naga::TypeInner::Struct { members, span };
                let r#type = ::naga::Type { name, inner };
                types.insert(r#type, Default::default())
            }

            fn shader_bytes(&self) -> Box<[u8]> {
                let mut data = vec![0u8; Self::SIZE];
                #field_bytes
                data.into_boxed_slice()
            }
        }

        impl #impl_generics #base_path::ShaderStruct for #struct_name #ty_generics #where_clause {
            const FIELDS: &'static [#base_path::ShaderField] = &[
                #field_info
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
