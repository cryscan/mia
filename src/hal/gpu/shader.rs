use half::f16;
use mia_derive::ShaderType;
use naga::{Handle, Scalar, Type, TypeInner, UniqueArena, VectorSize};

/// A GPU shader type.
/// A GPU shader type doesn't necessarily have the same layout as the corresponding Rust type.
/// For example, `[u32; 3]` in rust is aligned to 12 bytes, but in shader it's 16 bytes.
pub trait ShaderType {
    /// The size of the shader type in bytes.
    const SIZE: usize;
    /// The alignment of the shader type in bytes.
    const ALIGN: usize;

    /// Build the [`Type`] from this type. This adds dependent types to the arena as well.
    fn shader_type(types: &mut UniqueArena<Type>) -> Handle<Type>;
    /// Returns the index of a field in the shader type.
    fn shader_field_index(field: impl AsRef<str>) -> usize;
    /// Returns the offset in bytes of a field in the shader type.
    fn shader_field_offset(field: impl AsRef<str>) -> usize;
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

impl<T: ShaderScalar> ShaderType for T {
    const SIZE: usize = size_of::<Self>();
    const ALIGN: usize = align_of::<Self>();

    fn shader_type(types: &mut UniqueArena<Type>) -> Handle<Type> {
        let r#type = Type {
            name: None,
            inner: TypeInner::Scalar(T::shader_scalar()),
        };
        types.insert(r#type, Default::default())
    }

    fn shader_field_index(_: impl AsRef<str>) -> usize {
        panic!("scalar type does not have fields")
    }

    fn shader_field_offset(_: impl AsRef<str>) -> usize {
        panic!("scalar type does not have fields")
    }
}

macro_rules! impl_shader_type_vector {
    ($len:literal, $size:ident, $align:literal) => {
        impl<T: ShaderScalar> ShaderType for [T; $len] {
            const SIZE: usize = size_of::<Self>();
            const ALIGN: usize = $align * align_of::<T>();

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

            fn shader_field_index(field: impl AsRef<str>) -> usize {
                match field.as_ref() {
                    "x" => 0,
                    "y" => 1,
                    "z" => 2,
                    "w" => 3,
                    _ => panic!("invalid field name for vector type"),
                }
            }

            fn shader_field_offset(field: impl AsRef<str>) -> usize {
                match field.as_ref() {
                    "x" => 0,
                    "y" => <T as ShaderType>::ALIGN,
                    "z" => <T as ShaderType>::ALIGN * 2,
                    "w" => <T as ShaderType>::ALIGN * 3,
                    _ => panic!("invalid field name for vector type"),
                }
            }
        }
    };
}

impl_shader_type_vector!(2, Bi, 2);
impl_shader_type_vector!(3, Tri, 4);
impl_shader_type_vector!(4, Quad, 4);

macro_rules! impl_shader_type_matrix {
    ($m:literal, $n:literal, $vm:ident, $vn:ident, $($fields:ident),*) => {
        impl<T: ShaderScalar> ShaderType for [[T; $m]; $n] {
            const SIZE: usize = {
                #[derive(Default, ShaderType)]
                #[shader(crate = "crate", bound = "U: ShaderScalar")]
                #[allow(unused)]
                struct Inner<U> {
                    $($fields: [U; $m]),*
                }
                Inner::<T>::SIZE
            };
            const ALIGN: usize = <[T; $m] as ShaderType>::ALIGN;

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

            fn shader_field_index(field: impl AsRef<str>) -> usize {
                #[derive(Default, ShaderType)]
                #[shader(crate = "crate", bound = "U: ShaderScalar")]
                #[allow(unused)]
                struct Inner<U> {
                    $($fields: [U; $m]),*
                }
                Inner::<T>::shader_field_index(field)
            }

            fn shader_field_offset(field: impl AsRef<str>) -> usize {
                #[derive(Default, ShaderType)]
                #[shader(crate = "crate", bound = "U: ShaderScalar")]
                #[allow(unused)]
                struct Inner<U> {
                    $($fields: [U; $m]),*
                }
                Inner::<T>::shader_field_offset(field)
            }
        }
    };
}

impl_shader_type_matrix!(2, 2, Bi, Bi, x, y);
impl_shader_type_matrix!(2, 3, Bi, Tri, x, y, z);
impl_shader_type_matrix!(2, 4, Bi, Quad, x, y, z, w);
impl_shader_type_matrix!(3, 2, Tri, Bi, x, y);
impl_shader_type_matrix!(3, 3, Tri, Tri, x, y, z);
impl_shader_type_matrix!(3, 4, Tri, Quad, x, y, z, w);
impl_shader_type_matrix!(4, 2, Quad, Bi, x, y);
impl_shader_type_matrix!(4, 3, Quad, Tri, x, y, z);
impl_shader_type_matrix!(4, 4, Quad, Quad, x, y, z, w);

macro_rules! impl_shader_type_packed {
    ($type:ty, $inner:ty) => {
        impl ShaderType for $type {
            const SIZE: usize = <$inner>::SIZE;
            const ALIGN: usize = <$inner>::ALIGN;

            fn shader_type(types: &mut UniqueArena<Type>) -> Handle<Type> {
                <$inner>::shader_type(types)
            }

            fn shader_field_index(field: impl AsRef<str>) -> usize {
                <$inner>::shader_field_index(field)
            }

            fn shader_field_offset(field: impl AsRef<str>) -> usize {
                <$inner>::shader_field_offset(field)
            }
        }
    };
}

impl_shader_type_packed!(crate::loom::num::F16x4, [f16; 4]);
impl_shader_type_packed!(crate::loom::num::F32x4, [f32; 4]);
impl_shader_type_packed!(crate::loom::num::U4x8, u32);
impl_shader_type_packed!(crate::loom::num::U8x4, u32);

#[derive(Default, ShaderType)]
#[shader(crate = "crate")]
pub struct ShaderLayout {
    pub shape: [u32; 4],
    pub stride: [u32; 4],
}

impl From<crate::loom::layout::Layout> for ShaderLayout {
    fn from(value: crate::loom::layout::Layout) -> Self {
        let value = value.pad_to(4);
        let shape = value.shape().to_array().map(|x| x as u32);
        let stride = value.stride().to_array().map(|x| x as u32);
        Self { shape, stride }
    }
}

#[cfg(test)]
mod tests {
    use super::ShaderType;
    use half::f16;

    /// Checks size and alignment of basic shader types.
    #[test]
    fn test_builtin_shader_type() {
        // scalar types
        assert_eq!(<f16 as ShaderType>::SIZE, 2);
        assert_eq!(<f16 as ShaderType>::ALIGN, 2);
        assert_eq!(<f32 as ShaderType>::SIZE, 4);
        assert_eq!(<f32 as ShaderType>::ALIGN, 4);
        assert_eq!(<f64 as ShaderType>::SIZE, 8);
        assert_eq!(<f64 as ShaderType>::ALIGN, 8);

        // vector types
        assert_eq!(<[f32; 2] as ShaderType>::SIZE, 8);
        assert_eq!(<[f32; 2] as ShaderType>::ALIGN, 8);
        assert_eq!(<[f32; 3] as ShaderType>::SIZE, 12);
        assert_eq!(<[f32; 3] as ShaderType>::ALIGN, 16);
        assert_eq!(<[f32; 4] as ShaderType>::SIZE, 16);
        assert_eq!(<[f32; 4] as ShaderType>::ALIGN, 16);

        // vector type offsets
        assert_eq!(<[f32; 2] as ShaderType>::shader_field_offset("x"), 0);
        assert_eq!(<[f32; 2] as ShaderType>::shader_field_offset("y"), 4);
        assert_eq!(<[f32; 3] as ShaderType>::shader_field_offset("x"), 0);
        assert_eq!(<[f32; 3] as ShaderType>::shader_field_offset("y"), 4);
        assert_eq!(<[f32; 3] as ShaderType>::shader_field_offset("z"), 8);
        assert_eq!(<[f32; 4] as ShaderType>::shader_field_offset("x"), 0);
        assert_eq!(<[f32; 4] as ShaderType>::shader_field_offset("y"), 4);
        assert_eq!(<[f32; 4] as ShaderType>::shader_field_offset("z"), 8);
        assert_eq!(<[f32; 4] as ShaderType>::shader_field_offset("w"), 12);

        // matrix types
        assert_eq!(<[[f32; 2]; 2] as ShaderType>::SIZE, 16);
        assert_eq!(<[[f32; 2]; 2] as ShaderType>::ALIGN, 8);
        assert_eq!(<[[f32; 3]; 3] as ShaderType>::SIZE, 48);
        assert_eq!(<[[f32; 3]; 3] as ShaderType>::ALIGN, 16);
        assert_eq!(<[[f32; 4]; 4] as ShaderType>::SIZE, 64);
        assert_eq!(<[[f32; 4]; 4] as ShaderType>::ALIGN, 16);

        // matrix type offsets
        assert_eq!(<[[f32; 2]; 2] as ShaderType>::shader_field_offset("x"), 0);
        assert_eq!(<[[f32; 2]; 2] as ShaderType>::shader_field_offset("y"), 8);

        assert_eq!(<[[f32; 3]; 3] as ShaderType>::shader_field_offset("x"), 0);
        assert_eq!(<[[f32; 3]; 3] as ShaderType>::shader_field_offset("y"), 16);
        assert_eq!(<[[f32; 3]; 3] as ShaderType>::shader_field_offset("z"), 32);

        assert_eq!(<[[f32; 4]; 4] as ShaderType>::shader_field_offset("x"), 0);
        assert_eq!(<[[f32; 4]; 4] as ShaderType>::shader_field_offset("y"), 16);
        assert_eq!(<[[f32; 4]; 4] as ShaderType>::shader_field_offset("z"), 32);
        assert_eq!(<[[f32; 4]; 4] as ShaderType>::shader_field_offset("w"), 48);
    }

    #[test]
    fn test_derive_shader_type() {
        #[derive(Debug, Default, ShaderType)]
        #[shader(crate = "crate")]
        #[repr(C)]
        struct Data {
            color: [f32; 4],         // offset: 0, align: 16, size: 16
            position: [f32; 3],      // offset: 16, align: 16, size: 12
            rotation: [[f32; 3]; 3], // offset: 32, align: 16, size: 48
            #[shader(builtin = "index")]
            index: u32, // offset: 80, align: 4, size: 4
        }

        assert_eq!(Data::ALIGN, 16);
        assert_eq!(Data::SIZE, 96);

        assert_eq!(Data::shader_field_index("color"), 0);
        assert_eq!(Data::shader_field_index("position"), 1);
        assert_eq!(Data::shader_field_index("rotation"), 2);
        assert_eq!(Data::shader_field_index("index"), 3);

        assert_eq!(Data::shader_field_offset("color"), 0);
        assert_eq!(Data::shader_field_offset("position"), 16);
        assert_eq!(Data::shader_field_offset("rotation"), 32);
        assert_eq!(Data::shader_field_offset("index"), 80);

        let mut types = naga::UniqueArena::new();
        let ty = Data::shader_type(&mut types);
        let ty = types.get_handle(ty).expect("failed to find type");

        println!("{ty:#?}");
        assert_eq!(types.len(), 5);
    }
}
