use half::f16;
use naga::{Handle, Scalar, Type, TypeInner, UniqueArena, VectorSize};

pub trait ShaderType {
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
    ($len:literal, $size:ident) => {
        impl<T: ShaderScalar> ShaderType for [T; $len] {
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

impl_shader_type_vector!(2, Bi);
impl_shader_type_vector!(3, Tri);
impl_shader_type_vector!(4, Quad);

macro_rules! impl_shader_type_matrix {
    ($m:literal, $n:literal, $vm:ident, $vn:ident) => {
        impl<T: ShaderScalar> ShaderType for [[T; $m]; $n] {
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
