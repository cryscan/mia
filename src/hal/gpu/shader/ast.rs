use naga::{BinaryOperator, Handle, Literal, Scalar, UnaryOperator, UniqueArena, VectorSize};

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

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Vector {
    pub scalar: Scalar,
    pub size: VectorSize,
}

#[derive(Debug, PartialEq, Eq, Hash)]
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

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Array {
    pub r#type: Handle<Type>,
    pub size: ArraySize,
}

#[derive(Debug, PartialEq, Eq, Hash)]
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

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TypeAlias {
    pub name: Ident,
    pub r#type: Handle<Type>,
}

#[derive(Debug)]
pub struct Block {
    pub stmts: Vec<Statement>,
}

#[derive(Debug)]
pub enum Statement {
    LocalItem(LocalItem),
    Expression(Handle<Expression>),
    Block(Block),
    If(If),
    Return(Return),
}

#[derive(Debug)]
pub enum LocalItem {
    Let(Let),
    Var(Var),
}

#[derive(Debug)]
pub struct Local;

#[derive(Debug)]
pub struct Let {
    pub name: Ident,
    pub r#type: Handle<Type>,
    pub init: Handle<Expression>,
    pub local: Handle<Local>,
}

#[derive(Debug)]
pub struct Var {
    pub name: Ident,
    pub r#type: Handle<Type>,
    pub init: Handle<Expression>,
    pub local: Handle<Local>,
}

#[derive(Debug)]
pub struct If {
    pub condition: Handle<Expression>,
    pub accept: Block,
    pub reject: Block,
}

#[derive(Debug)]
pub struct Return {
    pub value: Option<Handle<Expression>>,
}

#[derive(Debug)]
pub enum Expression {
    Literal(Literal),
    Ident(Ident),
    Operator(Operator),
    Call(Call),
    Index(Index),
    Access(Access),
}

#[derive(Debug)]
pub enum Operator {
    Assign(Assign),
    Unary(Unary),
    Binary(Binary),
}

#[derive(Debug)]
pub struct Assign {
    pub place: Handle<Expression>,
    pub value: Handle<Expression>,
}

#[derive(Debug)]
pub struct Unary {
    pub op: UnaryOperator,
    pub expr: Handle<Expression>,
}

#[derive(Debug)]
pub struct Binary {
    pub op: BinaryOperator,
    pub lhs: Handle<Expression>,
    pub rhs: Handle<Expression>,
}

#[derive(Debug)]
pub struct Call {
    pub function: Ident,
    pub inputs: Vec<Handle<Expression>>,
}

#[derive(Debug)]
pub struct Index {
    pub base: Handle<Expression>,
    pub index: Handle<Expression>,
}

#[derive(Debug)]
pub struct Access {
    pub base: Handle<Expression>,
    pub field: Ident,
}
