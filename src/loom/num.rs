use bytemuck::{Pod, Zeroable};
use derive_more::{Deref, DerefMut, Display, From, Into};
use half::f16;
use u4::AsNibbles;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use wide::f32x4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DataType {
    F32,
    F16,
    U8,
    U16,
    U32,
    U4x8,
    U8x4,
    F32x4,
    F16x4,
}

impl DataType {
    /// Number of elements packed.
    pub const fn count(self) -> usize {
        match self {
            DataType::F32 => 1,
            DataType::F16 => 1,
            DataType::U8 => 1,
            DataType::U16 => 1,
            DataType::U32 => 1,
            DataType::U4x8 => 8,
            DataType::U8x4 => 4,
            DataType::F32x4 => 4,
            DataType::F16x4 => 4,
        }
    }

    /// Element Size in bytes.
    pub const fn size(self) -> usize {
        match self {
            DataType::F32 => size_of::<f32>(),
            DataType::F16 => size_of::<f16>(),
            DataType::U8 => size_of::<u8>(),
            DataType::U16 => size_of::<u16>(),
            DataType::U32 => size_of::<u32>(),
            DataType::U4x8 => size_of::<U4x8>(),
            DataType::U8x4 => size_of::<U8x4>(),
            DataType::F32x4 => size_of::<F32x4>(),
            DataType::F16x4 => size_of::<F16x4>(),
        }
    }
}

macro_rules! impl_bytemuck {
    ($ty:ty) => {
        unsafe impl ::bytemuck::Zeroable for $ty {}
        unsafe impl ::bytemuck::Pod for $ty {}
    };
}

macro_rules! impl_display {
    ($ty:ty) => {
        impl std::fmt::Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_list().entries(self.0.iter()).finish()
            }
        }
    };
}

#[derive(Debug, Default, Clone, Copy, Deref, DerefMut, From, Into)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[repr(C)]
pub struct U4x8(pub AsNibbles<[u8; 4]>);

impl From<u32> for U4x8 {
    fn from(value: u32) -> Self {
        Self(AsNibbles(value.to_le_bytes()))
    }
}

impl From<U4x8> for u32 {
    fn from(value: U4x8) -> Self {
        u32::from_le_bytes(value.0.0)
    }
}

#[derive(Debug, Default, Clone, Copy, Deref, DerefMut, From, Into)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[repr(C)]
pub struct U8x4(pub [u8; 4]);

impl From<u32> for U8x4 {
    fn from(value: u32) -> Self {
        Self(value.to_le_bytes())
    }
}

impl From<U8x4> for u32 {
    fn from(value: U8x4) -> Self {
        u32::from_le_bytes(value.0)
    }
}

#[derive(Debug, Default, Clone, Copy, Deref, DerefMut, From, Into)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[repr(C)]
pub struct F32x4(pub [f32; 4]);

impl From<f32x4> for F32x4 {
    fn from(value: f32x4) -> Self {
        Self(value.to_array())
    }
}

impl From<F32x4> for f32x4 {
    fn from(value: F32x4) -> Self {
        f32x4::new(value.0)
    }
}

impl F32x4 {
    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        let lhs = f32x4::from(self);
        let rhs = f32x4::from(other);
        (lhs * rhs).reduce_add()
    }
}

impl std::ops::Add for F32x4 {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let lhs = f32x4::from(self);
        let rhs = f32x4::from(other);
        Self::from(lhs + rhs)
    }
}

impl std::ops::Sub for F32x4 {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let lhs = f32x4::from(self);
        let rhs = f32x4::from(other);
        Self::from(lhs - rhs)
    }
}

impl std::ops::Mul for F32x4 {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        let lhs = f32x4::from(self);
        let rhs = f32x4::from(other);
        Self::from(lhs * rhs)
    }
}

impl std::ops::Div for F32x4 {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        let lhs = f32x4::from(self);
        let rhs = f32x4::from(other);
        Self::from(lhs / rhs)
    }
}

impl std::ops::AddAssign for F32x4 {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl std::ops::SubAssign for F32x4 {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl std::ops::MulAssign for F32x4 {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl std::ops::DivAssign for F32x4 {
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

#[derive(Debug, Default, Clone, Copy, Deref, DerefMut, From, Into)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[repr(C)]
pub struct F16x4(pub [f16; 4]);

impl F16x4 {
    #[inline]
    pub fn dot(self, other: Self) -> f16 {
        self[0] * other[0] + self[1] * other[1] + self[2] * other[2] + self[3] * other[3]
    }
}

impl std::ops::Add for F16x4 {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self([
            self[0] + other[0],
            self[1] + other[1],
            self[2] + other[2],
            self[3] + other[3],
        ])
    }
}

impl std::ops::Sub for F16x4 {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self([
            self[0] - other[0],
            self[1] - other[1],
            self[2] - other[2],
            self[3] - other[3],
        ])
    }
}

impl std::ops::Mul for F16x4 {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Self([
            self[0] * other[0],
            self[1] * other[1],
            self[2] * other[2],
            self[3] * other[3],
        ])
    }
}

impl std::ops::Div for F16x4 {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        Self([
            self[0] / other[0],
            self[1] / other[1],
            self[2] / other[2],
            self[3] / other[3],
        ])
    }
}

impl std::ops::AddAssign for F16x4 {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl std::ops::SubAssign for F16x4 {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl std::ops::MulAssign for F16x4 {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl std::ops::DivAssign for F16x4 {
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

impl_bytemuck!(U4x8);
impl_bytemuck!(U8x4);
impl_bytemuck!(F32x4);
impl_bytemuck!(F16x4);

impl_display!(U4x8);
impl_display!(U8x4);
impl_display!(F32x4);
impl_display!(F16x4);

pub trait Zero {
    fn zero() -> Self;
}

impl Zero for f32 {
    fn zero() -> Self {
        0.0
    }
}

impl Zero for f16 {
    fn zero() -> Self {
        Self::ZERO
    }
}

impl Zero for u8 {
    fn zero() -> Self {
        0
    }
}

impl Zero for u16 {
    fn zero() -> Self {
        0
    }
}

impl Zero for u32 {
    fn zero() -> Self {
        0
    }
}

impl Zero for U4x8 {
    fn zero() -> Self {
        Self(AsNibbles([0; 4]))
    }
}

impl Zero for U8x4 {
    fn zero() -> Self {
        Self([0; 4])
    }
}

impl Zero for F32x4 {
    fn zero() -> Self {
        Self([0.0; 4])
    }
}

impl Zero for F16x4 {
    fn zero() -> Self {
        Self([f16::ZERO; 4])
    }
}

pub trait One {
    fn one() -> Self;
}

impl One for f32 {
    fn one() -> Self {
        1.0
    }
}

impl One for f16 {
    fn one() -> Self {
        Self::ONE
    }
}

impl One for u8 {
    fn one() -> Self {
        1
    }
}

impl One for u16 {
    fn one() -> Self {
        1
    }
}

impl One for u32 {
    fn one() -> Self {
        1
    }
}

impl One for U4x8 {
    fn one() -> Self {
        Self(AsNibbles([0x11; 4]))
    }
}

impl One for U8x4 {
    fn one() -> Self {
        Self([1; 4])
    }
}

impl One for F32x4 {
    fn one() -> Self {
        Self([1.0; 4])
    }
}

impl One for F16x4 {
    fn one() -> Self {
        Self([f16::ONE; 4])
    }
}

pub trait Scalar: Sized + Zeroable + Pod + Zero + One + Send + Sync {
    const DATA_TYPE: DataType;
}

impl Scalar for f32 {
    const DATA_TYPE: DataType = DataType::F32;
}

impl Scalar for f16 {
    const DATA_TYPE: DataType = DataType::F16;
}

impl Scalar for u8 {
    const DATA_TYPE: DataType = DataType::U8;
}

impl Scalar for u16 {
    const DATA_TYPE: DataType = DataType::U16;
}

impl Scalar for u32 {
    const DATA_TYPE: DataType = DataType::U32;
}

impl Scalar for U4x8 {
    const DATA_TYPE: DataType = DataType::U4x8;
}

impl Scalar for U8x4 {
    const DATA_TYPE: DataType = DataType::U8x4;
}

impl Scalar for F32x4 {
    const DATA_TYPE: DataType = DataType::F32x4;
}

impl Scalar for F16x4 {
    const DATA_TYPE: DataType = DataType::F16x4;
}

pub trait Float:
    Scalar
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
{
}

impl Float for f32 {}
impl Float for f16 {}

pub trait Float4:
    Scalar
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
{
    type Element: Float;

    #[inline]
    fn index(index: usize) -> usize {
        index * Self::DATA_TYPE.count()
    }
}

impl Float4 for F32x4 {
    type Element = f32;
}
impl Float4 for F16x4 {
    type Element = f16;
}

#[cfg(test)]
mod tests {
    use crate::loom::num::{U4x8, U8x4};

    #[test]
    pub fn test_packed_types() {
        println!("{}", U4x8::from(0xfeed2025));
        println!("{}", U8x4::from(0xfeed2025));
    }
}
