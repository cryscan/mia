use half::f16;
use naga::{Handle, Scalar, Type, TypeInner, UniqueArena};

/// The type can be convert to an IR node in the shader.
pub trait ShaderType {
    /// Converts this type into its shader IR.
    fn shader_type(types: &mut UniqueArena<Type>) -> Handle<Type>;
}

macro_rules! impl_shader_type_scalar {
    ($type:ty, $scalar:ident) => {
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
