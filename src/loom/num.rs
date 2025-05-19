use bytemuck::{Pod, Zeroable};
use derive_more::{Deref, DerefMut, Display, From, Into};
use half::f16;
use serde::{Deserialize, Serialize};
use u4::AsNibbles;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display, Serialize, Deserialize)]
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
            DataType::U4x8 => size_of::<u32>(),
            DataType::U8x4 => size_of::<u32>(),
            DataType::F32x4 => size_of::<[f32; 4]>(),
            DataType::F16x4 => size_of::<[f16; 4]>(),
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

#[derive(Debug, Default, Clone, Copy, Deref, DerefMut, From, Into, Serialize, Deserialize)]
#[serde(transparent)]
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

#[derive(Debug, Default, Clone, Copy, Deref, DerefMut, From, Into, Serialize, Deserialize)]
#[serde(transparent)]
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

#[derive(Debug, Default, Clone, Copy, Deref, DerefMut, From, Into, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(C)]
pub struct F32x4(pub [f32; 4]);

#[derive(Debug, Default, Clone, Copy, Deref, DerefMut, From, Into, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(C)]
pub struct F16x4(pub [f16; 4]);

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

pub trait Float: Scalar {}

impl Float for f32 {}
impl Float for f16 {}

pub trait Float4: Scalar {}

impl Float4 for F32x4 {}
impl Float4 for F16x4 {}

#[cfg(test)]
mod tests {
    use crate::loom::num::{U4x8, U8x4};

    #[test]
    pub fn test_packed_types() {
        println!("{}", U4x8::from(0xfeed2025));
        println!("{}", U8x4::from(0xfeed2025));
    }
}
