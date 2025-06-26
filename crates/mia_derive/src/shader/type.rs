use half::f16;
use naga::{Expression, Literal, Module, Scalar, Type, TypeInner, VectorSize};

/// The type can be convert to an IR node in the shader.
pub trait ShaderType<T> {
    /// Converts this type into its shader IR.
    fn to_shader_ir() -> T;
}

macro_rules! impl_shader_type_scalar {
    ($type:ty, $scalar:ident) => {
        impl ShaderType<Scalar> for $type {
            fn to_shader_ir() -> Scalar {
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

impl<T> ShaderType<TypeInner> for T
where
    T: ShaderType<Scalar>,
{
    fn to_shader_ir() -> TypeInner {
        TypeInner::Scalar(Self::to_shader_ir())
    }
}

impl<T> ShaderType<Type> for T
where
    T: ShaderType<Scalar>,
{
    fn to_shader_ir() -> Type {
        let name = None;
        let inner = <Self as ShaderType<TypeInner>>::to_shader_ir();
        Type { name, inner }
    }
}

pub struct Vector<T, const N: usize>([T; N]);

macro_rules! impl_shader_type_vector {
    ($size:literal, $vector_size:expr) => {
        impl<T> ShaderType<TypeInner> for Vector<T, $size>
        where
            T: ShaderType<Scalar>,
        {
            fn to_shader_ir() -> TypeInner {
                TypeInner::Vector {
                    scalar: T::to_shader_ir(),
                    size: $vector_size,
                }
            }
        }

        impl<T> ShaderType<Type> for Vector<T, $size>
        where
            T: ShaderType<Scalar>,
        {
            fn to_shader_ir() -> Type {
                let name = None;
                let inner = Self::to_shader_ir();
                Type { name, inner }
            }
        }
    };
}

impl_shader_type_vector!(2, VectorSize::Bi);
impl_shader_type_vector!(3, VectorSize::Tri);
impl_shader_type_vector!(4, VectorSize::Quad);

pub struct Matrix<T, const M: usize, const N: usize>([[T; N]; M]);

macro_rules! impl_shader_type_matrix {
    ($m:literal, $n:literal, $vm:expr, $vn:expr) => {
        impl<T> ShaderType<TypeInner> for Matrix<T, $m, $n>
        where
            T: ShaderType<Scalar>,
        {
            fn to_shader_ir() -> TypeInner {
                TypeInner::Matrix {
                    columns: $vm,
                    rows: $vn,
                    scalar: T::to_shader_ir(),
                }
            }
        }

        impl<T> ShaderType<Type> for Matrix<T, $m, $n>
        where
            T: ShaderType<Scalar>,
        {
            fn to_shader_ir() -> Type {
                let name = None;
                let inner = Self::to_shader_ir();
                Type { name, inner }
            }
        }
    };
}

impl_shader_type_matrix!(2, 2, VectorSize::Bi, VectorSize::Bi);
impl_shader_type_matrix!(2, 3, VectorSize::Bi, VectorSize::Tri);
impl_shader_type_matrix!(2, 4, VectorSize::Bi, VectorSize::Quad);
impl_shader_type_matrix!(3, 2, VectorSize::Tri, VectorSize::Bi);
impl_shader_type_matrix!(3, 3, VectorSize::Tri, VectorSize::Tri);
impl_shader_type_matrix!(3, 4, VectorSize::Tri, VectorSize::Quad);
impl_shader_type_matrix!(4, 2, VectorSize::Quad, VectorSize::Bi);
impl_shader_type_matrix!(4, 3, VectorSize::Quad, VectorSize::Tri);
impl_shader_type_matrix!(4, 4, VectorSize::Quad, VectorSize::Quad);

/// The type's value can be convert to an IR node in the shader.
pub trait ShaderValue<T> {
    /// Converts this value into its shader IR.
    fn to_shader_ir(&self, module: &Module) -> T;
}

macro_rules! impl_shader_value_literal {
    ($type:ty, $literal:ident) => {
        impl ShaderValue<Literal> for $type {
            fn to_shader_ir(&self, _: &Module) -> Literal {
                Literal::$literal(*self)
            }
        }
    };
}

impl_shader_value_literal!(f16, F16);
impl_shader_value_literal!(f32, F32);
impl_shader_value_literal!(f64, F64);
impl_shader_value_literal!(u32, U32);
impl_shader_value_literal!(u64, U64);
impl_shader_value_literal!(i32, I32);
impl_shader_value_literal!(i64, I64);
impl_shader_value_literal!(bool, Bool);

impl<T> ShaderValue<Expression> for T
where
    T: ShaderValue<Literal>,
{
    fn to_shader_ir(&self, module: &Module) -> Expression {
        Expression::Literal(self.to_shader_ir(module))
    }
}
