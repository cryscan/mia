use std::{marker::PhantomData, sync::Arc};

use derive_more::{Deref, DerefMut};
use thiserror::Error;

use super::{
    device::Device,
    layout::{IntoLayout, Layout},
    num::{DataType, Scalar},
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
    #[error("tensor slice error: slice {1} is not compatible with layout {0}")]
    Slice(Layout, Slice),
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
struct TensorId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorUntyped<D: Device> {
    device: Arc<D>,
    layout: Layout,
    r#type: DataType,
    id: uid::Id<TensorId>,
}

impl<D: Device> TensorUntyped<D> {
    #[inline]
    pub fn layout(&self) -> Layout {
        self.layout.clone()
    }

    #[inline]
    pub fn data_type(&self) -> DataType {
        self.r#type
    }

    /// Converts the untyped type to a typed one. Returns error if type mismatches.
    #[inline]
    pub fn try_into_typed<T: Scalar>(self) -> Result<Tensor<D, T>, TensorError> {
        if self.r#type != T::DATA_TYPE {
            return Err(TensorError::Type(self.r#type, T::DATA_TYPE));
        }
        Ok(Tensor {
            tensor: self,
            phantom: PhantomData,
        })
    }

    #[inline]
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.device)
    }
}

/// A statically typed tensor. Good to fit into typed APIs.
#[derive(Debug, Clone, PartialEq, Eq, Deref, DerefMut)]
pub struct Tensor<D: Device, T: Scalar> {
    #[deref]
    #[deref_mut]
    tensor: TensorUntyped<D>,
    phantom: PhantomData<T>,
}

impl<D: Device, T: Scalar> Tensor<D, T> {
    /// Transform the tensor into an untyped one.
    #[inline]
    pub fn into_untyped(self) -> TensorUntyped<D> {
        self.tensor
    }

    /// Create a tensor of zeros.
    #[inline]
    pub fn zeros(device: D, layout: impl IntoLayout) -> Self {
        let device = Arc::new(device);
        let layout = layout.into_layout();
        let r#type = T::DATA_TYPE;
        let id = uid::Id::new();
        let phantom = PhantomData;
        let tensor = TensorUntyped {
            device,
            layout,
            r#type,
            id,
        };
        Self { tensor, phantom }
    }

    /// Reshape the tensor, leaving the underlying data untouched.
    #[inline]
    pub fn reshape(mut self, layout: Layout) -> Result<Self, TensorError> {
        if self.layout.size() != layout.size() {
            return Err(TensorError::Reshape(self.layout(), layout));
        }
        self.layout = layout;
        self.id = uid::Id::new();
        Ok(self)
    }
}
