use proc_macro2::TokenStream;
use quote::quote;
use syn::{Attribute, DeriveInput, LitStr, Path, spanned::Spanned};

pub fn derive_shader_type(input: DeriveInput) -> TokenStream {
    // parse shader_type attributes
    let mut crate_name = None;
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

    let struct_name = &input.ident;

    // retrieve struct field information
    let fields = match &input.data {
        syn::Data::Struct(data_struct) => &data_struct.fields,
        _ => {
            return syn::Error::new(input.span(), "`ShaderType` can only be derived for structs")
                .to_compile_error();
        }
    };

    let members: TokenStream = fields
        .iter()
        .map(|field| {
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
            let offset = quote!(::bytemuck::offset_of!(#struct_name, #field_name) as u32);
            quote! {
                members.push(::naga::StructMember {
                    name: #name,
                    ty: #ty,
                    binding: #binding,
                    offset: #offset,
                });
            }
        })
        .collect();

    let name = struct_name.to_string();
    quote! {
        impl #base_path::ShaderType for #struct_name {
            fn shader_type(types: &mut ::naga::UniqueArena<::naga::Type>) -> ::naga::Handle<::naga::Type> {
                let name = Some(#name.to_string());
                let members = {
                    let mut members = Vec::new();
                    #members;
                    members
                };
                let span = ::std::mem::size_of::<#struct_name>() as u32;
                let inner = ::naga::TypeInner::Struct { members, span };
                let r#type = ::naga::Type { name, inner };
                types.insert(r#type, Default::default())
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
