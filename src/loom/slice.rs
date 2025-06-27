use std::sync::Arc;

use casey::snake;
use derive_more::{Deref, DerefMut, Display, From, Into};
use itertools::Itertools;

use super::{device::Device, layout::Layout, num::Scalar, tensor::Tensor};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Display)]
pub enum Axis {
    #[default]
    #[display("..")]
    Full,
    #[display("{_0}")]
    One(usize),
}

impl From<usize> for Axis {
    #[inline]
    fn from(value: usize) -> Self {
        Self::One(value)
    }
}

impl From<std::ops::RangeFull> for Axis {
    #[inline]
    fn from(_: std::ops::RangeFull) -> Self {
        Self::Full
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Deref, DerefMut, From, Into, Display)]
#[display("[{}]", _0.iter().format(", "))]
pub struct Slice(Arc<[Axis]>);

impl From<Vec<Axis>> for Slice {
    #[inline]
    fn from(value: Vec<Axis>) -> Self {
        Self(value.into())
    }
}

macro_rules! impl_slice_from {
    ($t:ident) => {
        impl<$t: Into<Axis>> From<$t> for Slice {
            #[inline]
            fn from(snake!($t): $t) -> Self {
                Self([snake!($t).into()].into())
            }
        }
    };
    ($($t:ident),+) => {
        impl<$($t),+> From<($($t),+)> for Slice
        where
            $($t: Into<Axis>),+
        {
            #[inline]
            fn from(($(snake!($t)),+): ($($t),+)) -> Self {
                Self([$(snake!($t).into()),+].into())
            }
        }
    };
}

impl_slice_from!(T0);
impl_slice_from!(T0, T1);
impl_slice_from!(T0, T1, T2);
impl_slice_from!(T0, T1, T2, T3);
impl_slice_from!(T0, T1, T2, T3, T4);
impl_slice_from!(T0, T1, T2, T3, T4, T5);
impl_slice_from!(T0, T1, T2, T3, T4, T5, T6);
impl_slice_from!(T0, T1, T2, T3, T4, T5, T6, T7);

impl Slice {
    /// Creates a full slice of the same mode as a `Layout`.
    #[inline]
    pub fn from_layout(layout: Layout) -> Self {
        Self::from(vec![Axis::Full; layout.len()])
    }

    /// Returns `true` if the slice contains only full axes.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.iter().all(|&axis| matches!(axis, Axis::Full))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deref, DerefMut)]
pub struct TensorSlice<D, T> {
    #[deref]
    #[deref_mut]
    tensor: Tensor<D, T>,
    slice: Slice,
}

impl<D: Device, T: Scalar> TensorSlice<D, T> {
    #[inline]
    pub fn slice(&self) -> Slice {
        self.slice.clone()
    }
}

impl<D: Device, T: Scalar> Tensor<D, T> {
    /// Create a [`TensorSlice`] from the tensor.
    ///
    /// # Panics
    /// This method will panic if the slice length does not match the tensor layout length,
    /// or if the slice contains out-of-bounds indices.
    #[inline]
    pub fn slice(self, slice: Slice) -> TensorSlice<D, T> {
        let layout = self.layout();
        let shape = layout.shape();
        assert_eq!(
            layout.len(),
            slice.len(),
            "slice length must match tensor layout length"
        );
        assert!(
            itertools::izip!(slice.iter(), shape.iter())
                .filter_map(|(&axis, &shape)| match axis {
                    Axis::Full => None,
                    Axis::One(index) => Some((index, shape)),
                })
                .all(|(index, shape)| index < shape),
            "slice contains out-of-bounds indices: {slice:?} for shape {shape:?}",
        );
        let tensor = self;
        TensorSlice { tensor, slice }
    }
}
