use half::f16;
use itertools::Itertools;
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
    /// Convert `self` to bytes in shader.
    fn shader_bytes(&self) -> Box<[u8]>;
}

/// A trait for types that can be used as scalar values in shaders.
///
/// This trait is implemented for primitive numeric types that can be directly used in shaders,
/// such as `f16`, `f32`, `f64`, `u32`, `u64`, `i32`, `i64`, and `bool`.
///
/// Types implementing this trait must also implement [`ShaderType`].
pub trait ShaderScalar: ShaderType + crate::loom::num::Scalar {
    /// Returns the shader scalar type information.
    fn shader_scalar() -> Scalar;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderField {
    pub name: Option<&'static str>,
    pub offset: usize,
    pub size: usize,
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

    fn shader_field(name: &str) -> Option<(usize, &'static ShaderField)> {
        Self::FIELDS
            .iter()
            .find_position(|field| field.name == Some(name))
    }
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
// impl_shader_type_scalar!(bool, BOOL);

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

    fn shader_bytes(&self) -> Box<[u8]> {
        let mut data = vec![0u8; Self::SIZE];
        data[..size_of::<Self>()].copy_from_slice(bytemuck::bytes_of(self));
        data.into_boxed_slice()
    }
}

macro_rules! impl_shader_type_vector {
    ($len:literal, $size:ident, $align:literal, $($fields:ident),*) => {
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

            fn shader_bytes(&self) -> Box<[u8]> {
                let mut data = vec![0u8; Self::SIZE];
                data[..size_of::<Self>()].copy_from_slice(bytemuck::bytes_of(self));
                data.into_boxed_slice()
            }
        }

        impl<T: ShaderScalar> ShaderStruct for [T; $len] {
            const FIELDS: &'static [ShaderField] = {
                #[derive(Default, ShaderType)]
                #[shader(crate = "crate", bound = "T: ShaderScalar")]
                #[allow(unused)]
                struct Inner<T> {
                    $($fields: T),*
                }
                Inner::<T>::FIELDS
            };
        }
    };
}

impl_shader_type_vector!(2, Bi, 2, x, y);
impl_shader_type_vector!(3, Tri, 4, x, y, z);
impl_shader_type_vector!(4, Quad, 4, x, y, z, w);

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

            fn shader_bytes(&self) -> Box<[u8]> {
                let mut data = vec![0u8; Self::SIZE];
                for (index, field) in Self::FIELDS.iter().enumerate() {
                    let field_data = self[index].shader_bytes();
                    let start = field.offset;
                    let end = start + std::cmp::min(field_data.len(), field.size);
                    data[start..end].copy_from_slice(&field_data[..end - start]);
                }
                data.into_boxed_slice()
            }
        }

        impl<T: ShaderScalar> ShaderStruct for [[T; $m]; $n] {
            const FIELDS: &'static [ShaderField] = {
                #[derive(Default, ShaderType)]
                #[shader(crate = "crate", bound = "T: ShaderScalar")]
                #[allow(unused)]
                struct Inner<T> {
                    $($fields: [T; $m]),*
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

            fn shader_bytes(&self) -> Box<[u8]> {
                let mut data = vec![0u8; Self::SIZE];
                data[..size_of::<Self>()].copy_from_slice(bytemuck::bytes_of(self));
                data.into_boxed_slice()
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

    fn check_field<T: ShaderStruct>(index: usize, name: &'static str, offset: usize, size: usize) {
        let name = Some(name);
        assert_eq!(T::FIELDS[index], ShaderField { name, offset, size });
    }

    fn field_span<T: ShaderStruct>(index: usize) -> std::ops::Range<usize> {
        let field = T::FIELDS[index];
        field.offset..field.offset + field.size
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

        check_field::<[f32; 2]>(0, "x", 0, 4);
        check_field::<[f32; 2]>(1, "y", 4, 4);
        check_field::<[f32; 3]>(0, "x", 0, 4);
        check_field::<[f32; 3]>(1, "y", 4, 4);
        check_field::<[f32; 3]>(2, "z", 8, 4);
        check_field::<[f32; 4]>(0, "x", 0, 4);
        check_field::<[f32; 4]>(1, "y", 4, 4);
        check_field::<[f32; 4]>(2, "z", 8, 4);
        check_field::<[f32; 4]>(3, "w", 12, 4);

        // matrix types
        assert_eq!(<[[f32; 2]; 2]>::SIZE, 16);
        assert_eq!(<[[f32; 2]; 2]>::ALIGN, 8);
        assert_eq!(<[[f32; 3]; 3]>::SIZE, 48);
        assert_eq!(<[[f32; 3]; 3]>::ALIGN, 16);
        assert_eq!(<[[f32; 4]; 4]>::SIZE, 64);
        assert_eq!(<[[f32; 4]; 4]>::ALIGN, 16);

        // matrix type offsets
        check_field::<[[f32; 2]; 2]>(0, "x", 0, 8);
        check_field::<[[f32; 2]; 2]>(1, "y", 8, 8);

        check_field::<[[f32; 3]; 3]>(0, "x", 0, 16);
        check_field::<[[f32; 3]; 3]>(1, "y", 16, 16);
        check_field::<[[f32; 3]; 3]>(2, "z", 32, 16);

        check_field::<[[f32; 4]; 4]>(0, "x", 0, 16);
        check_field::<[[f32; 4]; 4]>(1, "y", 16, 16);
        check_field::<[[f32; 4]; 4]>(2, "z", 32, 16);
        check_field::<[[f32; 4]; 4]>(3, "w", 48, 16);
    }

    #[test]
    fn test_derive_shader_type() {
        #[derive(Debug, Default, Clone, ShaderType)]
        #[shader(crate = "crate")]
        #[repr(C)]
        struct Data {
            color: [f32; 4],         // offset: 0, align: 16, size: 16
            position: [f32; 3],      // offset: 16, align: 16, size: 16
            rotation: [[f32; 3]; 3], // offset: 32, align: 16, size: 48
            #[shader(builtin = "index")]
            index: u32, // offset: 80, align: 4, size: 4
        }

        assert_eq!(Data::ALIGN, 16);
        assert_eq!(Data::SIZE, 96);

        check_field::<Data>(0, "color", 0, 16);
        check_field::<Data>(1, "position", 16, 16);
        check_field::<Data>(2, "rotation", 32, 48);
        check_field::<Data>(3, "index", 80, 4);

        let mut types = naga::UniqueArena::new();
        let ty = Data::shader_type(&mut types);
        let ty = types.get_handle(ty).expect("failed to find type");

        println!("{ty:#?}");
        assert_eq!(types.len(), 5);
    }

    #[test]
    fn test_derive_shader_bytes() {
        #[derive(Debug, Default, Clone, ShaderType)]
        #[shader(crate = "crate")]
        #[repr(C)]
        struct SimpleStruct {
            a: f32,
            b: [f32; 2],
            c: u32,
        }

        let data = SimpleStruct {
            a: 1.5,
            b: [2.0, 3.0],
            c: 42,
        };

        println!("{:#?}", SimpleStruct::FIELDS);

        let bytes = data.shader_bytes();
        assert_eq!(bytes.len(), SimpleStruct::SIZE);

        // test that all fields can be serialized individually
        let a_bytes = data.a.shader_bytes();
        let b_bytes = data.b.shader_bytes();
        let c_bytes = data.c.shader_bytes();

        assert_eq!(a_bytes.len(), 4);
        assert_eq!(b_bytes.len(), 8);
        assert_eq!(c_bytes.len(), 4);

        assert_eq!(&bytes[field_span::<SimpleStruct>(0)], &a_bytes[..]);
        assert_eq!(&bytes[field_span::<SimpleStruct>(1)], &b_bytes[..]);
        assert_eq!(&bytes[field_span::<SimpleStruct>(2)], &c_bytes[..]);
    }

    #[test]
    fn test_derive_shader_bytes_nested() {
        #[derive(Debug, Default, Clone, ShaderType)]
        #[shader(crate = "crate")]
        #[repr(C)]
        struct Inner {
            x: f32,
            y: f32,
        }

        #[derive(Debug, Default, Clone, ShaderType)]
        #[shader(crate = "crate")]
        #[repr(C)]
        struct Outer {
            z: u32,
            inner: Inner,
        }

        let data = Outer {
            inner: Inner { x: 1.0, y: 2.0 },
            z: 100,
        };

        let bytes = data.shader_bytes();
        assert_eq!(bytes.len(), Outer::SIZE);

        let z_bytes = data.z.shader_bytes();
        let inner_bytes = data.inner.shader_bytes();

        assert_eq!(&bytes[field_span::<Outer>(0)], &z_bytes[..]);
        assert_eq!(&bytes[field_span::<Outer>(1)], &inner_bytes[..]);
    }

    #[test]
    fn test_derive_shader_bytes_tuple_struct() {
        #[derive(Debug, Default, Clone, ShaderType)]
        #[shader(crate = "crate")]
        #[repr(C)]
        struct TupleStruct(f32, [f32; 3], u32);

        let data = TupleStruct(5.0, [1.0, 2.0, 3.0], 255);
        let bytes = data.shader_bytes();
        assert_eq!(bytes.len(), TupleStruct::SIZE);

        // test that all fields can be serialized individually
        let field0_bytes = data.0.shader_bytes();
        let field1_bytes = data.1.shader_bytes();
        let field2_bytes = data.2.shader_bytes();

        assert_eq!(field0_bytes.len(), 4);
        assert_eq!(field1_bytes.len(), 16); // [f32; 3] has shader size 16 due to alignment
        assert_eq!(field2_bytes.len(), 4);

        assert_eq!(&bytes[field_span::<TupleStruct>(0)], &field0_bytes[..]);
        assert_eq!(&bytes[field_span::<TupleStruct>(1)], &field1_bytes[..]);
        assert_eq!(&bytes[field_span::<TupleStruct>(2)], &field2_bytes[..]);
    }

    #[test]
    fn test_derive_shader_bytes_matrix() {
        #[derive(Debug, Default, Clone, ShaderType)]
        #[shader(crate = "crate")]
        #[repr(C)]
        struct MatrixStruct {
            transform: [[f32; 3]; 3],
            scale: f32,
        }

        let data = MatrixStruct {
            transform: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            scale: 2.0,
        };

        let bytes = data.shader_bytes();
        assert_eq!(bytes.len(), MatrixStruct::SIZE);

        // test that all fields can be serialized individually
        let matrix_bytes = data.transform.shader_bytes();
        let scale_bytes = data.scale.shader_bytes();

        assert_eq!(matrix_bytes.len(), 48); // [[f32; 3]; 3] has shader size 48
        assert_eq!(scale_bytes.len(), 4);

        assert_eq!(&bytes[field_span::<MatrixStruct>(0)], &matrix_bytes[..]);
        assert_eq!(&bytes[field_span::<MatrixStruct>(1)], &scale_bytes[..]);
    }
}
