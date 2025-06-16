use std::{marker::PhantomData, sync::Arc};

use derive_more::Display;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{
    device::Device,
    layout::{IntoLayout, Layout},
    num::Scalar,
    ops::{Access, TensorIr, TensorOp, TensorTape, ZeroOp},
};

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

    /// Returns a mutable reference to the tensor tape.
    ///
    /// ## Panics
    /// This method will panic if the tape is not unique, meaning that there are other references to the same tape.
    #[inline]
    pub fn tape_mut(&mut self) -> &mut TensorTape {
        Arc::get_mut(&mut self.tape).expect("tape is not unique")
    }

    /// Returns the element count of the tensor.
    #[inline]
    pub fn data_count(&self) -> usize {
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
    ///
    /// ## Panics
    /// This method will panic if the new layout's data size is greater than the buffer size.
    #[inline]
    pub fn reshape(mut self, layout: impl IntoLayout) -> Self {
        let layout = layout.into_layout();
        self.layout = layout;
        assert!(
            self.data_size() <= self.buffer_size(),
            "data size must be less than or equal to buffer size"
        );
        self
    }

    /// Re-interpret the tensor as one of another type, leaving the underlying data untouched.
    ///
    /// ## Panics
    /// This method will panic if the new tensor's data size is greater than the buffer size.
    #[inline]
    pub fn cast<U: Scalar>(self, layout: impl IntoLayout) -> Tensor<D, U> {
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
        assert!(
            output.data_size() <= output.buffer_size(),
            "data size must be less than or equal to buffer size"
        );
        output
    }
}

impl<D: Device + Clone, T: Scalar> Tensor<D, T> {
    /// Create a tensor of zeros.
    ///
    /// # Panics
    /// This method will panic if the data size is greater than the buffer size.
    #[inline]
    pub fn zeros(device: D, layout: impl IntoLayout, size: usize) -> Self {
        let layout = layout.into_layout();
        let id = TensorId(uuid::Uuid::new_v4());
        let ir = unsafe { TensorIr::unique::<T>(id, layout.clone(), Access::WriteOnly) };
        let ops: Vec<Box<dyn TensorOp>> = vec![Box::new(ZeroOp::new(ir))];
        let tape = Arc::new(TensorTape { id, ops });
        let phantom = PhantomData;
        let output = Self {
            device,
            layout,
            id,
            size,
            tape,
            phantom,
        };
        assert!(
            output.data_size() <= output.buffer_size(),
            "zeros requires the data size to be less than or equal to the buffer size"
        );
        output
    }

    /// Create a tensor of zeros from current device and layout.
    #[inline]
    pub fn zeros_like(&self) -> Self {
        let device = self.device.clone();
        let layout = self.layout();
        let size = self.buffer_size();
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
    }
}
