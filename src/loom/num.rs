use bytemuck::{Pod, Zeroable};
use derive_more::Display;
use half::f16;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display, Serialize, Deserialize)]
pub enum DataType {
    F32,
    F16,
    U8,
    U16,
    U32,
    PackedU4x8,
    PackedU8x4,
    PackedF32x4,
    PackedF16x4,
}

impl DataType {
    /// Returns number of element packed in this data type.
    pub const fn count(self) -> usize {
        match self {
            DataType::F32 => 1,
            DataType::F16 => 1,
            DataType::U8 => 1,
            DataType::U16 => 1,
            DataType::U32 => 1,
            DataType::PackedU4x8 => 8,
            DataType::PackedU8x4 => 4,
            DataType::PackedF32x4 => 4,
            DataType::PackedF16x4 => 4,
        }
    }
}

macro_rules! impl_bytemuck {
    ($ty:ty) => {
        unsafe impl ::bytemuck::Zeroable for $ty {}
        unsafe impl ::bytemuck::Pod for $ty {}
    };
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct PackedU4x8(pub u32);

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct PackedU8x4(pub u32);

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct PackedF32x4(pub [f32; 4]);

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct PackedF16x4(pub [f16; 4]);

impl_bytemuck!(PackedU4x8);
impl_bytemuck!(PackedU8x4);
impl_bytemuck!(PackedF32x4);
impl_bytemuck!(PackedF16x4);

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

impl Zero for PackedU4x8 {
    fn zero() -> Self {
        Self(0)
    }
}

impl Zero for PackedU8x4 {
    fn zero() -> Self {
        Self(0)
    }
}

impl Zero for PackedF32x4 {
    fn zero() -> Self {
        Self([0.0; 4])
    }
}

impl Zero for PackedF16x4 {
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

impl One for PackedU4x8 {
    fn one() -> Self {
        Self(0x11111111)
    }
}

impl One for PackedU8x4 {
    fn one() -> Self {
        Self(0x01010101)
    }
}

impl One for PackedF32x4 {
    fn one() -> Self {
        Self([1.0; 4])
    }
}

impl One for PackedF16x4 {
    fn one() -> Self {
        Self([f16::ONE; 4])
    }
}

pub trait Scalar: Sized + Zeroable + Pod + Zero + One + Send + Sync {
    const DATA_TYPE: DataType;
}

pub trait Float: Scalar {}
pub trait PackedFloat4: Scalar {}

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

impl Scalar for PackedU4x8 {
    const DATA_TYPE: DataType = DataType::PackedU4x8;
}

impl Scalar for PackedU8x4 {
    const DATA_TYPE: DataType = DataType::PackedU8x4;
}

impl Scalar for PackedF32x4 {
    const DATA_TYPE: DataType = DataType::PackedF32x4;
}

impl Scalar for PackedF16x4 {
    const DATA_TYPE: DataType = DataType::PackedF16x4;
}

impl Float for f32 {}
impl Float for f16 {}

impl PackedFloat4 for PackedF32x4 {}
impl PackedFloat4 for PackedF16x4 {}
