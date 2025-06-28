use half::f16;
use mia_derive::ShaderType;
use naga::{Handle, Scalar, Type, TypeInner, UniqueArena, VectorSize};

/// A GPU shader type.
///
/// Note that it doesn't necessarily have the same layout as the corresponding Rust type.
/// For example, `[u32; 3]` in rust is aligned to 12 bytes, but in shader it's 16 bytes.
pub trait ShaderType {
    /// The size of the shader type in bytes.
    const SIZE: usize;
    /// The alignment of the shader type in bytes.
    const ALIGN: usize;

    /// Build the [`Type`] from this type. This adds dependent types to the arena as well.
    fn shader_type(types: &mut UniqueArena<Type>) -> Handle<Type>;
}

/// A trait for types that can be used as scalar values in shaders.
///
/// This trait is implemented for primitive numeric types that can be directly used in shaders,
/// such as `f16`, `f32`, `f64`, `u32`, `u64`, `i32`, `i64`, and `bool`.
///
/// Types implementing this trait must also implement [`ShaderType`].
pub trait ShaderScalar: ShaderType {
    /// Returns the shader scalar type information.
    fn shader_scalar() -> Scalar;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderField {
    pub name: Option<&'static str>,
    pub index: usize,
    pub offset: usize,
}

/// A trait for shader types that have a struct-like memory layout with named fields.
///
/// This trait provides methods to query the memory layout of fields within the type when used in shaders.
/// It is implemented for vectors, matrices, and user-defined structs that are used in shaders.
///
/// Types implementing this trait must also implement [`ShaderType`].
pub trait ShaderStruct: ShaderType {
    /// Returns the fields of the shader type.
    const FIELDS: &'static [ShaderField];
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
}

macro_rules! impl_shader_type_vector {
    ($len:literal, $size:ident, $align:literal) => {
        impl<T: ShaderScalar> ShaderType for [T; $len] {
            const SIZE: usize = size_of::<Self>().div_ceil(Self::ALIGN) * Self::ALIGN;
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
        }

        impl<T: ShaderScalar> ShaderStruct for [T; $len] {
            const FIELDS: &'static [ShaderField] = &[
                ShaderField {
                    name: Some("x"),
                    index: 0,
                    offset: 0,
                },
                ShaderField {
                    name: Some("y"),
                    index: 1,
                    offset: <T as ShaderType>::SIZE,
                },
                ShaderField {
                    name: Some("z"),
                    index: 2,
                    offset: <T as ShaderType>::SIZE * 2,
                },
                ShaderField {
                    name: Some("w"),
                    index: 3,
                    offset: <T as ShaderType>::SIZE * 3,
                },
            ];
        }
    };
}

impl_shader_type_vector!(2, Bi, 2);
impl_shader_type_vector!(3, Tri, 4);
impl_shader_type_vector!(4, Quad, 4);

macro_rules! impl_shader_type_matrix {
    ($m:literal, $n:literal, $vm:ident, $vn:ident, $($fields:ident),*) => {
        impl<T: ShaderScalar> ShaderType for [[T; $m]; $n] {
            const SIZE: usize = $n * <[T; $n] as ShaderType>::SIZE;
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
        }

        impl<T: ShaderScalar> ShaderStruct for [[T; $m]; $n] {
            const FIELDS: &'static [ShaderField] = {
                #[derive(Default, ShaderType)]
                #[shader(crate = "crate", bound = "U: ShaderScalar")]
                #[allow(unused)]
                struct Inner<U> {
                    $($fields: [U; $m]),*
                }
                Inner::<T>::FIELDS
            };
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
        }
    };
}

impl_shader_type_packed!(crate::loom::num::F16x4, [f16; 4]);
impl_shader_type_packed!(crate::loom::num::F32x4, [f32; 4]);
impl_shader_type_packed!(crate::loom::num::U4x8, u32);
impl_shader_type_packed!(crate::loom::num::U8x4, u32);

#[derive(Debug, Default, Clone, ShaderType)]
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

#[derive(Debug, Default, Clone, ShaderType)]
#[shader(crate = "crate")]
pub struct ShaderLayoutBundle {
    pub block: ShaderLayout,
    pub thread: ShaderLayout,
    pub custom: ShaderLayout,
}

impl From<super::LayoutBundle> for ShaderLayoutBundle {
    fn from(value: super::LayoutBundle) -> Self {
        let block = value.block.into();
        let thread = value.thread.into();
        let custom = value.custom.into();
        Self {
            block,
            thread,
            custom,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ShaderField, ShaderStruct, ShaderType};
    use half::f16;

    fn check_field<T: ShaderStruct>(index: usize, name: Option<&'static str>, offset: usize) {
        assert_eq!(
            T::FIELDS[index],
            ShaderField {
                name,
                index,
                offset,
            }
        );
    }

    /// Checks size and alignment of basic shader types.
    #[test]
    fn test_builtin_shader_type() {
        // scalar types
        assert_eq!(<f16>::SIZE, 2);
        assert_eq!(<f16>::ALIGN, 2);
        assert_eq!(<f32>::SIZE, 4);
        assert_eq!(<f32>::ALIGN, 4);
        assert_eq!(<f64>::SIZE, 8);
        assert_eq!(<f64>::ALIGN, 8);

        // vector types
        assert_eq!(<[f32; 2]>::SIZE, 8);
        assert_eq!(<[f32; 2]>::ALIGN, 8);
        assert_eq!(<[f32; 3]>::SIZE, 16);
        assert_eq!(<[f32; 3]>::ALIGN, 16);
        assert_eq!(<[f32; 4]>::SIZE, 16);
        assert_eq!(<[f32; 4]>::ALIGN, 16);

        check_field::<[f32; 2]>(0, Some("x"), 0);
        check_field::<[f32; 2]>(1, Some("y"), 4);
        check_field::<[f32; 3]>(0, Some("x"), 0);
        check_field::<[f32; 3]>(1, Some("y"), 4);
        check_field::<[f32; 3]>(2, Some("z"), 8);
        check_field::<[f32; 4]>(0, Some("x"), 0);
        check_field::<[f32; 4]>(1, Some("y"), 4);
        check_field::<[f32; 4]>(2, Some("z"), 8);
        check_field::<[f32; 4]>(3, Some("w"), 12);

        // matrix types
        assert_eq!(<[[f32; 2]; 2]>::SIZE, 16);
        assert_eq!(<[[f32; 2]; 2]>::ALIGN, 8);
        assert_eq!(<[[f32; 3]; 3]>::SIZE, 48);
        assert_eq!(<[[f32; 3]; 3]>::ALIGN, 16);
        assert_eq!(<[[f32; 4]; 4]>::SIZE, 64);
        assert_eq!(<[[f32; 4]; 4]>::ALIGN, 16);

        // matrix type offsets
        check_field::<[[f32; 2]; 2]>(0, Some("x"), 0);
        check_field::<[[f32; 2]; 2]>(1, Some("y"), 8);

        check_field::<[[f32; 3]; 3]>(0, Some("x"), 0);
        check_field::<[[f32; 3]; 3]>(1, Some("y"), 16);
        check_field::<[[f32; 3]; 3]>(2, Some("z"), 32);

        check_field::<[[f32; 4]; 4]>(0, Some("x"), 0);
        check_field::<[[f32; 4]; 4]>(1, Some("y"), 16);
        check_field::<[[f32; 4]; 4]>(2, Some("z"), 32);
        check_field::<[[f32; 4]; 4]>(3, Some("w"), 48);
    }

    #[test]
    fn test_derive_shader_type() {
        #[derive(Debug, Default, Clone, ShaderType)]
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

        check_field::<Data>(0, Some("color"), 0);
        check_field::<Data>(1, Some("position"), 16);
        check_field::<Data>(2, Some("rotation"), 32);
        check_field::<Data>(3, Some("index"), 80);

        let mut types = naga::UniqueArena::new();
        let ty = Data::shader_type(&mut types);
        let ty = types.get_handle(ty).expect("failed to find type");

        println!("{ty:#?}");
        assert_eq!(types.len(), 5);
    }
}
