use naga::{Arena, Handle, Scalar, VectorSize};

pub struct Module {
    pub types: Arena<Type>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ident {
    pub name: &'static str,
}

#[derive(Debug)]
pub enum Type {
    Scalar(Scalar),
    Vector(Vector),
    Matrix(Matrix),
    Array(Array),
    Struct(Struct),
}

#[derive(Debug)]
pub struct Vector {
    pub r#type: Handle<Type>,
    pub size: VectorSize,
}

#[derive(Debug)]
pub struct Matrix {
    pub r#type: Handle<Type>,
    pub rows: VectorSize,
    pub columns: VectorSize,
}

#[derive(Debug)]
pub enum ArraySize {
    Dynamic,
    Constant(std::num::NonZeroU32),
}

#[derive(Debug)]
pub struct Array {
    pub r#type: Handle<Type>,
    pub size: ArraySize,
}

#[derive(Debug)]
pub struct StructField {
    pub name: Ident,
    pub r#type: Handle<Type>,
    pub offset: u32,
}

#[derive(Debug)]
pub struct Struct {
    pub name: Ident,
    pub fields: Vec<StructField>,
}

#[derive(Debug)]
pub struct TypeAlias {
    pub name: Ident,
    pub r#type: Handle<Type>,
}
