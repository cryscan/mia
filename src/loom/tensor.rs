use std::{marker::PhantomData, sync::Arc};

use derive_more::Display;
use thiserror::Error;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{
    device::Device,
    layout::{IntoLayout, Layout, LayoutError, Shape},
    num::Scalar,
    ops::{Access, TensorIr, TensorOp, TensorTape, ZeroOp},
};

#[derive(Debug, Error)]
pub enum TensorErrorKind {
    #[error("unmatched layout: {0}")]
    UnexpectedLayout(Layout),
    #[error("incompatible layouts: expected {expected}, found {actual}")]
    IncompatibleLayouts { expected: Layout, actual: Layout },
    #[error("incompatible shapes: expected {expected}, found {actual}")]
    IncompatibleShapes { expected: Shape, actual: Shape },
    #[error("insufficient buffer size: required {required} bytes, available {available} bytes")]
    InsufficientBufferSize { required: usize, available: usize },
    #[error("unaligned data type: expected {expected}, found {actual}")]
    UnalignedDataType { expected: usize, actual: usize },
    #[error("unique tape required but found multiple references")]
    NonUniqueTape,
    #[error("layout error: {0}")]
    Layout(#[from] LayoutError),
}

#[derive(Debug, Display)]
#[display("{error}\n\nBacktrace:\n{backtrace}")]
pub struct TensorError {
    pub error: TensorErrorKind,
    pub backtrace: std::backtrace::Backtrace,
}

impl From<TensorErrorKind> for TensorError {
    fn from(error: TensorErrorKind) -> Self {
        let backtrace = std::backtrace::Backtrace::capture();
        Self { error, backtrace }
    }
}

impl From<LayoutError> for TensorError {
    fn from(error: LayoutError) -> Self {
        TensorErrorKind::from(error).into()
    }
}

impl std::error::Error for TensorError {
    fn cause(&self) -> Option<&dyn std::error::Error> {
        Some(&self.error)
    }

    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

#[derive(Debug, Default, Display, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[repr(transparent)]
pub struct TensorId(pub uuid::Uuid);

/// A statically typed tensor. Good to fit into typed APIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor<D, T> {
    /// The device this tensor is bound to.
    device: D,
    /// The layout of the tensor, which describes its shape and strides.
    layout: Layout,
    /// The unique identifier of the tensor.
    id: TensorId,
    /// The size of the tensor in bytes.
    size: usize,
    /// The tape of operations that have been applied to this tensor.
    tape: Arc<TensorTape>,
    /// Phantom data to ensure the tensor is statically typed.
    phantom: PhantomData<T>,
}

impl<D: Device, T: Scalar> Tensor<D, T> {
    /// Returns a reference to the device this tensor is bound to.
    #[inline]
    pub fn device(&self) -> &D {
        &self.device
    }

    /// Returns the layout of the tensor.
    #[inline]
    pub fn layout(&self) -> Layout {
        self.layout.clone()
    }

    /// Returns the unique identifier of the tensor.
    #[inline]
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Returns a reference to the tensor tape.
    #[inline]
    pub fn tape(&self) -> &TensorTape {
        &self.tape
    }

    /// Returns a mutable reference to the tensor tape. Returns error if the tape is not unique.
    #[inline]
    pub fn try_tape_mut(&mut self) -> Result<&mut TensorTape, TensorError> {
        Arc::get_mut(&mut self.tape)
            .ok_or(TensorErrorKind::NonUniqueTape)
            .map_err(Into::into)
    }

    /// Returns a mutable reference to the tensor tape.
    ///
    /// ## Panics
    /// This method will panic if the tape is not unique, meaning that there are other references to the same tape.
    #[inline]
    pub fn tape_mut(&mut self) -> &mut TensorTape {
        self.try_tape_mut().expect("tape is not unique")
    }

    /// Returns the element count of the tensor.
    #[inline]
    pub fn data_len(&self) -> usize {
        self.layout.co_size() * T::DATA_TYPE.count()
    }

    /// Returns the size of the tensor data in bytes.
    #[inline]
    pub fn data_size(&self) -> usize {
        self.layout.co_size() * size_of::<T>()
    }

    /// Returns the size of the buffer in bytes. It should be always greater than or equal to the data size.
    #[inline]
    pub fn buffer_size(&self) -> usize {
        self.size
    }

    /// Returns the reference count of the tensor, which indicates how many references to the tensor exist.
    #[inline]
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.tape)
    }

    /// Reshape the tensor, leaving the underlying data untouched.
    #[inline]
    pub fn reshape(mut self, layout: impl IntoLayout) -> Result<Self, TensorError> {
        let layout = layout.into_layout();
        self.layout = layout;
        self.check_size().map_err(Into::into)
    }

    /// Re-interpret the tensor as one of another type, leaving the underlying data untouched.
    #[inline]
    pub fn cast<U: Scalar>(self, layout: impl IntoLayout) -> Result<Tensor<D, U>, TensorError> {
        let layout = layout.into_layout();
        let device = self.device;
        let id = self.id;
        let size = self.size;
        let tape = self.tape;
        let phantom = PhantomData;
        let output = Tensor {
            device,
            layout,
            id,
            size,
            tape,
            phantom,
        };
        output.check_size()?.check_align::<U>().map_err(Into::into)
    }
}

impl<D: Device, T: Scalar> Tensor<D, T> {
    /// Checks if the data types have exact the same alignment.
    #[inline]
    pub fn check_align<U: Scalar>(self) -> Result<Self, TensorErrorKind> {
        let expected = align_of::<T>();
        let actual = align_of::<U>();
        let err = || TensorErrorKind::UnalignedDataType { expected, actual };
        (expected == actual).then_some(self).ok_or_else(err)
    }

    /// Checks if `self` has enough buffer size available.
    #[inline]
    pub fn check_size(self) -> Result<Self, TensorErrorKind> {
        let required = self.data_size();
        let available = self.buffer_size();
        let err = || TensorErrorKind::InsufficientBufferSize {
            required,
            available,
        };
        (required <= available).then_some(self).ok_or_else(err)
    }

    /// Checks if the tensor's dimension is in the given range.
    #[inline]
    pub fn check_dim<R>(self, expected: R) -> Result<Self, TensorErrorKind>
    where
        R: std::ops::RangeBounds<usize>,
    {
        let len = self.layout.len();
        let layout = self.layout();
        let err = || TensorErrorKind::UnexpectedLayout(layout);
        (expected.contains(&len)).then_some(self).ok_or_else(err)
    }

    /// Checks if the layout of `self` matches the reference. Skips 0-sized modes.
    #[inline]
    pub fn check_layout(self, r#ref: impl IntoLayout) -> Result<Self, TensorErrorKind> {
        let layout = self.layout();
        let r#ref = r#ref.into_layout();
        let err = {
            let expected = r#ref.clone();
            let actual = layout.clone();
            || TensorErrorKind::IncompatibleLayouts { expected, actual }
        };
        layout
            .iter()
            .zip(r#ref.iter())
            .filter(|(_, (x, _))| *x != 0)
            .all(|(x, y)| x == y)
            .then_some(self)
            .ok_or_else(err)
    }

    /// Checks if the shape of `self` matches the reference. Skips 0 modes.
    #[inline]
    pub fn check_shape(self, r#ref: impl Into<Shape>) -> Result<Self, TensorErrorKind> {
        let shape = self.layout.shape();
        let r#ref: Shape = r#ref.into();
        let err = {
            let expected = r#ref.clone();
            let actual = shape.clone();
            || TensorErrorKind::IncompatibleShapes { expected, actual }
        };
        shape
            .iter()
            .zip(r#ref.iter())
            .filter(|(_, x)| **x != 0)
            .all(|(x, y)| x == y)
            .then_some(self)
            .ok_or_else(err)
    }
}

impl<D: Device + Clone, T: Scalar> Tensor<D, T> {
    /// Create a tensor with the given device, layout, and buffer size.
    #[inline]
    pub fn init(device: D, layout: impl IntoLayout, size: usize) -> Result<Self, TensorError> {
        let layout = layout.into_layout();
        let id = TensorId(uuid::Uuid::new_v4());

        let ir = unsafe { TensorIr::unique::<T>(id, layout.clone(), Access::WriteOnly) };
        let ops: Vec<Box<dyn TensorOp>> = vec![Box::new(ZeroOp::new(ir))];
        let tape = Arc::new(TensorTape { id, ops });

        let phantom = PhantomData;
        Self {
            device,
            layout,
            id,
            size,
            tape,
            phantom,
        }
        .check_size()
        .map_err(Into::into)
    }

    /// Create a tensor with the given device, layout, and buffer size.
    #[inline]
    pub fn zeros(device: D, layout: impl IntoLayout) -> Self {
        let layout = layout.into_layout();
        let size = layout.co_size() * size_of::<T>();
        Self::init(device, layout, size).expect("failed to initialize tensor")
    }

    /// Create a tensor of zeros from current device and layout with minimum buffer size.
    #[inline]
    pub fn zeros_like<U: Scalar>(&self) -> Tensor<D, U> {
        let device = self.device.clone();
        let layout = self.layout();
        Tensor::zeros(device, layout)
    }
}
