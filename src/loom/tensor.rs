use std::{marker::PhantomData, sync::Arc};

use derive_more::{AsMut, AsRef, Display};
use thiserror::Error;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{
    device::Device,
    layout::{IntoLayout, Layout},
    num::{DataType, Scalar},
    ops::{Access, TensorIr, TensorOp, TensorTape, ZeroOp},
    slice::Slice,
};

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("tensor type error: data type {0} mismatches {1}")]
    Type(DataType, DataType),
    #[error("tensor creation error: layout {0}'s size not match data len {1}")]
    Create(Layout, usize),
    #[error("tensor reshape error: layout {0}'s size not match layout {1}'s")]
    Reshape(Layout, Layout),
    #[error("tensor cast error: data size before casting ({0}) not match that after ({1})")]
    Cast(usize, usize),
    #[error("tensor slice error: slice {1} is not compatible with layout {0}")]
    Slice(Layout, Slice),
    #[error("tensor layout error: expected {0}, got {1}")]
    Layout(Layout, Layout),
}

#[derive(Debug, Default, Display, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[repr(transparent)]
pub struct TensorId(pub uuid::Uuid);

/// A statically typed tensor. Good to fit into typed APIs.
#[derive(Debug, Clone, PartialEq, Eq, AsRef, AsMut)]
pub struct Tensor<D, T> {
    device: Arc<D>,
    layout: Layout,
    id: TensorId,
    #[as_ref]
    #[as_mut]
    tape: Arc<TensorTape>,
    phantom: PhantomData<T>,
}

impl<D: Device, T: Scalar> Tensor<D, T> {
    #[inline]
    pub fn device(&self) -> &D {
        &self.device
    }

    #[inline]
    pub fn layout(&self) -> Layout {
        self.layout.clone()
    }

    #[inline]
    pub fn id(&self) -> TensorId {
        self.id
    }

    #[inline]
    pub fn tape(&self) -> &TensorTape {
        &self.tape
    }

    #[inline]
    pub fn data_count(&self) -> usize {
        self.layout.size() * T::DATA_TYPE.count()
    }

    #[inline]
    pub fn data_size(&self) -> usize {
        self.layout.size() * size_of::<T>()
    }

    #[inline]
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.tape)
    }

    /// Reshape the tensor, leaving the underlying data untouched.
    #[inline]
    pub fn reshape(mut self, layout: impl IntoLayout) -> Result<Self, TensorError> {
        let layout = layout.into_layout();
        if self.layout.size() != layout.size() {
            return Err(TensorError::Reshape(self.layout, layout));
        }
        self.layout = layout;
        Ok(self)
    }

    /// Re-interpret the tensor as another type, leaving the underlying data untouched.
    #[inline]
    pub fn cast<U: Scalar>(self, layout: impl IntoLayout) -> Result<Tensor<D, U>, TensorError> {
        let layout = layout.into_layout();
        let size = layout.size() * U::DATA_TYPE.count();
        if self.data_size() != size {
            return Err(TensorError::Cast(self.data_size(), size));
        }
        let device = self.device;
        let id = self.id;
        let tape = self.tape;
        let phantom = PhantomData;
        Ok(Tensor {
            device,
            layout,
            id,
            tape,
            phantom,
        })
    }
}

impl<D: Device, T: Scalar> Tensor<D, T> {
    /// Create a tensor of zeros.
    #[inline]
    pub fn zeros(device: Arc<D>, layout: impl IntoLayout) -> Self {
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
            tape,
            phantom,
        }
    }

    /// Create a tensor of zeros from current device and layout.
    #[inline]
    pub fn zeros_like(&self) -> Self {
        let device = self.device.clone();
        let layout = self.layout();
        let id = TensorId(uuid::Uuid::new_v4());
        let ir = unsafe { TensorIr::unique::<T>(id, layout.clone(), Access::WriteOnly) };
        let ops: Vec<Box<dyn TensorOp>> = vec![Box::new(ZeroOp::new(ir))];
        let tape = Arc::new(TensorTape { id, ops });
        let phantom = PhantomData;
        Self {
            device,
            layout,
            id,
            tape,
            phantom,
        }
    }
}
