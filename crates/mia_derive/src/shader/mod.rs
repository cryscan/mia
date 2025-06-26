use half::f16;
use proc_macro2::TokenStream;
use quote::quote;
use rustc_hash::FxHashMap as HashMap;
use syn::{parse_quote, spanned::Spanned};

use r#type::ShaderType;

mod r#type;

type TypeMap = HashMap<syn::Type, (naga::Type, usize)>;
type TypeHandleMap = HashMap<syn::Type, (naga::Handle<naga::Type>, usize)>;

pub fn shader(input: syn::ItemFn) -> Result<TokenStream, syn::Error> {
    let mut module = naga::Module::default();

    let _types = collect_types(&mut module, &input.block.stmts)?;

    Ok(quote! {})
}

fn shader_ir<T: ShaderType<naga::Type>>() -> naga::Type {
    T::to_shader_ir()
}

pub trait SpanToNaga {
    fn to_naga(&self) -> naga::Span;
}

impl SpanToNaga for proc_macro2::Span {
    fn to_naga(&self) -> naga::Span {
        let range = self.byte_range();
        naga::Span::new(range.start as u32, range.end as u32)
    }
}

/// Scan the stmts and collect all types used in the shader.
/// Add all types to the module and return a map of type to handle and offset.
fn collect_types(
    module: &mut naga::Module,
    stmts: &[syn::Stmt],
) -> Result<TypeHandleMap, syn::Error> {
    let mut builtin_types = TypeMap::default();
    builtin_types.insert(parse_quote!(f16), (shader_ir::<f16>(), size_of::<f16>()));
    builtin_types.insert(parse_quote!(f32), (shader_ir::<f32>(), size_of::<f32>()));
    builtin_types.insert(parse_quote!(u32), (shader_ir::<u32>(), size_of::<u32>()));
    builtin_types.insert(parse_quote!(u64), (shader_ir::<u64>(), size_of::<u64>()));
    builtin_types.insert(parse_quote!(i32), (shader_ir::<i32>(), size_of::<i32>()));
    builtin_types.insert(parse_quote!(i64), (shader_ir::<i64>(), size_of::<i64>()));
    builtin_types.insert(parse_quote!(bool), (shader_ir::<bool>(), size_of::<bool>()));

    let mut types = TypeHandleMap::default();
    for stmt in stmts {
        match stmt {
            syn::Stmt::Local(_local) => todo!(),
            syn::Stmt::Item(item) => {
                if let syn::Item::Struct(item_struct) = item {
                    if item_struct.fields.is_empty() {
                        return Err(syn::Error::new_spanned(
                            item_struct,
                            "struct must have at least one field",
                        ));
                    }

                    let mut offset = 0;
                    let mut members = Vec::new();
                    for field in &item_struct.fields {
                        let binding = parse_field_binding(&field.attrs)?;
                        if let Some(&(ty, size)) = types.get(&field.ty) {
                            let name = field.ident.as_ref().map(ToString::to_string);
                            members.push(naga::StructMember {
                                name,
                                ty,
                                offset,
                                binding,
                            });
                            offset += size as u32;
                        } else if let Some((ty, size)) = builtin_types.get(&field.ty).cloned() {
                            let ty = module.types.insert(ty, field.span().to_naga());
                            let name = field.ident.as_ref().map(ToString::to_string);
                            members.push(naga::StructMember {
                                name,
                                ty,
                                offset,
                                binding,
                            });
                            offset += size as u32;
                        } else {
                            return Err(syn::Error::new_spanned(field, "unsupported type"));
                        }
                    }

                    let span = offset;
                    let inner = naga::TypeInner::Struct { members, span };
                    let name = Some(item_struct.ident.to_string());
                    let ty = naga::Type { inner, name };
                    let handle = module
                        .types
                        .insert(ty.clone(), item_struct.span().to_naga());
                    let ident = &item_struct.ident;
                    types.insert(parse_quote!(#ident), (handle, offset as usize));
                }
            }
            syn::Stmt::Expr(_expr, _semi) => todo!(),
            syn::Stmt::Macro(_stmt_macro) => todo!(),
        }
    }
    Ok(types)
}

fn parse_field_binding(attrs: &[syn::Attribute]) -> Result<Option<naga::Binding>, syn::Error> {
    for attr in attrs {
        if attr.path().is_ident("builtin") {
            let builtin_name: syn::Ident = attr.parse_args()?;
            let builtin = match builtin_name.to_string().as_str() {
                "position" => naga::BuiltIn::Position { invariant: false },
                "position_invariant" => naga::BuiltIn::Position { invariant: true },
                "view_index" => naga::BuiltIn::ViewIndex,
                // vertex
                "base_instance" => naga::BuiltIn::BaseInstance,
                "base_vertex" => naga::BuiltIn::BaseVertex,
                "clip_distance" => naga::BuiltIn::ClipDistance,
                "cull_distance" => naga::BuiltIn::CullDistance,
                "instance_index" => naga::BuiltIn::InstanceIndex,
                "point_size" => naga::BuiltIn::PointSize,
                "vertex_index" => naga::BuiltIn::VertexIndex,
                "draw_id" => naga::BuiltIn::DrawID,
                // fragment
                "frag_depth" => naga::BuiltIn::FragDepth,
                "point_coord" => naga::BuiltIn::PointCoord,
                "front_facing" => naga::BuiltIn::FrontFacing,
                "primitive_index" => naga::BuiltIn::PrimitiveIndex,
                "sample_index" => naga::BuiltIn::SampleIndex,
                "sample_mask" => naga::BuiltIn::SampleMask,
                // compute
                "global_invocation_id" => naga::BuiltIn::GlobalInvocationId,
                "local_invocation_id" => naga::BuiltIn::LocalInvocationId,
                "local_invocation_index" => naga::BuiltIn::LocalInvocationIndex,
                "workgroup_id" => naga::BuiltIn::WorkGroupId,
                "workgroup_size" => naga::BuiltIn::WorkGroupSize,
                "num_workgroups" => naga::BuiltIn::NumWorkGroups,
                // subgroup
                "num_subgroups" => naga::BuiltIn::NumSubgroups,
                "subgroup_id" => naga::BuiltIn::SubgroupId,
                "subgroup_size" => naga::BuiltIn::SubgroupSize,
                "subgroup_invocation_id" => naga::BuiltIn::SubgroupInvocationId,
                _ => return Err(syn::Error::new_spanned(&builtin_name, "unknown builtin")),
            };
            return Ok(Some(naga::Binding::BuiltIn(builtin)));
        } else if attr.path().is_ident("location") {
            let location: syn::LitInt = attr.parse_args()?;
            let location_value = location.base10_parse::<u32>()?;
            return Ok(Some(naga::Binding::Location {
                location: location_value,
                interpolation: None,
                sampling: None,
                blend_src: None,
            }));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_types() {
        let input: syn::ItemFn = parse_quote! {
            fn main() {
                struct Vertex {
                    #[builtin(position)]
                    position: f32,
                    #[builtin(vertex_index)]
                    vertex_index: u32,
                }
            }
        };
        let mut module = naga::Module::default();
        let types = collect_types(&mut module, &input.block.stmts).unwrap();
        assert_eq!(types.len(), 1);
        assert_eq!(module.types.len(), 3);
    }
}
