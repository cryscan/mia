use naga::{Handle, Scalar, UniqueArena, VectorSize};

#[derive(Debug, Default)]
pub struct Module {
    pub types: UniqueArena<Type>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ident {
    pub name: &'static str,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Scalar(Scalar),
    Vector(Vector),
    Matrix(Matrix),
    Array(Array),
    Struct(Struct),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Vector {
    pub scalar: Scalar,
    pub size: VectorSize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]

pub struct Matrix {
    pub scalar: Scalar,
    pub rows: VectorSize,
    pub columns: VectorSize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArraySize {
    #[default]
    Dynamic,
    Constant(std::num::NonZeroU32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Array {
    pub r#type: Handle<Type>,
    pub size: ArraySize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructField {
    pub name: Option<Ident>,
    pub r#type: Handle<Type>,
    pub offset: u32,
}

#[derive(Debug)]
pub struct Struct {
    pub name: Ident,
    pub fields: Vec<StructField>,
}

impl PartialEq for Struct {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Struct {}

impl std::hash::Hash for Struct {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeAlias {
    pub name: Ident,
    pub r#type: Handle<Type>,
}
