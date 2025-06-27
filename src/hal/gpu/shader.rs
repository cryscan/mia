use half::f16;
use naga::{Handle, Scalar, Type, TypeInner, UniqueArena, VectorSize};

/// A GPU shader type.
/// A GPU shader type doesn't necessarily have the same layout as the corresponding Rust type.
/// For example, `[u32; 3]` in rust is aligned to 12 bytes, but in shader it's 16 bytes.
pub trait ShaderType: Sized {
    /// The size of the shader type in bytes.
    const SIZE: usize = size_of::<Self>();
    /// The alignment of the shader type in bytes.
    const ALIGN: usize = align_of::<Self>();

    fn shader_type(types: &mut UniqueArena<Type>) -> Handle<Type>;
}

pub trait ShaderScalar {
    fn shader_scalar() -> Scalar;
}

macro_rules! impl_shader_type_scalar {
    ($type:ty, $scalar:ident) => {
        impl ShaderScalar for $type {
            fn shader_scalar() -> Scalar {
                Scalar::$scalar
            }
        }
        impl ShaderType for $type {
            fn shader_type(types: &mut UniqueArena<Type>) -> Handle<Type> {
                let r#type = Type {
                    name: None,
                    inner: TypeInner::Scalar(Scalar::$scalar),
                };
                types.insert(r#type, Default::default())
            }
        }
    };
}

impl_shader_type_scalar!(f16, F16);
impl_shader_type_scalar!(f32, F32);
impl_shader_type_scalar!(f64, F64);
impl_shader_type_scalar!(u32, U32);
impl_shader_type_scalar!(u64, U64);
impl_shader_type_scalar!(i32, I32);
impl_shader_type_scalar!(i64, I64);
impl_shader_type_scalar!(bool, BOOL);

macro_rules! impl_shader_type_vector {
    ($len:literal, $size:ident, $align:expr) => {
        impl<T: ShaderScalar> ShaderType for [T; $len] {
            const ALIGN: usize = $align;

            fn shader_type(types: &mut UniqueArena<Type>) -> Handle<Type> {
                let r#type = Type {
                    name: None,
                    inner: TypeInner::Vector {
                        size: VectorSize::$size,
                        scalar: T::shader_scalar(),
                    },
                };
                types.insert(r#type, Default::default())
            }
        }
    };
}

impl_shader_type_vector!(2, Bi, align_of::<[T; 2]>());
impl_shader_type_vector!(3, Tri, align_of::<[T; 4]>());
impl_shader_type_vector!(4, Quad, align_of::<[T; 4]>());

macro_rules! impl_shader_type_matrix {
    ($m:literal, $n:literal, $vm:ident, $vn:ident) => {
        impl<T: ShaderScalar> ShaderType for [[T; $m]; $n] {
            const SIZE: usize = $n * <[T; $m] as ShaderType>::SIZE;
            const ALIGN: usize = $n * <[T; $m] as ShaderType>::ALIGN;

            fn shader_type(types: &mut UniqueArena<Type>) -> Handle<Type> {
                let r#type = Type {
                    name: None,
                    inner: TypeInner::Matrix {
                        columns: VectorSize::$vn,
                        rows: VectorSize::$vm,
                        scalar: T::shader_scalar(),
                    },
                };
                types.insert(r#type, Default::default())
            }
        }
    };
}

impl_shader_type_matrix!(2, 2, Bi, Bi);
impl_shader_type_matrix!(2, 3, Bi, Tri);
impl_shader_type_matrix!(2, 4, Bi, Quad);
impl_shader_type_matrix!(3, 2, Tri, Bi);
impl_shader_type_matrix!(3, 3, Tri, Tri);
impl_shader_type_matrix!(3, 4, Tri, Quad);
impl_shader_type_matrix!(4, 2, Quad, Bi);
impl_shader_type_matrix!(4, 3, Quad, Tri);
impl_shader_type_matrix!(4, 4, Quad, Quad);

impl ShaderType for crate::loom::layout::Layout {
    /// In shader, we treat [`Layout`](crate::loom::layout::Layout) as `vec4<u32>`.
    fn shader_type(types: &mut UniqueArena<Type>) -> Handle<Type> {
        <[u32; 4]>::shader_type(types)
    }
}

#[cfg(test)]
mod tests {
    use super::ShaderType;
    use mia_derive::ShaderType;

    #[test]
    fn test_derive_shader_type() {
        #[derive(Debug, Default, ShaderType)]
        #[shader(crate = "crate")]
        #[repr(C)]
        struct Data {
            color: [f32; 4],
            position: [f32; 3],
            #[shader(builtin = "index")]
            index: u32,
        }

        let mut types = naga::UniqueArena::new();
        let ty = Data::shader_type(&mut types);
        println!("{:#?}", types.get_handle(ty).expect("failed to find type"));
        assert_eq!(types.len(), 4);
    }
}
