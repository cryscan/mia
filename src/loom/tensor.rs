use std::{marker::PhantomData, sync::Arc};

use derive_more::{Deref, DerefMut};
use serde::{Deserialize, Serialize};
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
    #[error("tensor cast error: data size before casting ({0}) not match that after ({1})")]
    Cast(usize, usize),
    #[error("tensor slice error: slice {1} is not compatible with layout {0}")]
    Slice(Layout, Slice),
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorUntyped<D> {
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

    #[inline]
    pub fn data_count(&self) -> usize {
        self.layout.size() * self.r#type.count()
    }

    #[inline]
    pub fn data_size(&self) -> usize {
        self.layout.size() * self.r#type.size()
    }

    #[inline]
    pub fn id(&self) -> uid::Id<TensorId> {
        self.id
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

impl<D: Device + Clone> TensorUntyped<D> {
    #[inline]
    pub fn device(&self) -> D {
        self.device.as_ref().clone()
    }
}

/// A statically typed tensor. Good to fit into typed APIs.
#[derive(Debug, Clone, PartialEq, Eq, Deref, DerefMut)]
pub struct Tensor<D, T> {
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

    #[inline]
    pub fn data_count(&self) -> usize {
        self.layout.size() * T::DATA_TYPE.count()
    }

    #[inline]
    pub fn data_size(&self) -> usize {
        self.layout.size() * size_of::<T>()
    }

    /// Reshape the tensor, leaving the underlying data untouched.
    #[inline]
    pub fn reshape(mut self, layout: impl IntoLayout) -> Result<Self, TensorError> {
        let layout = layout.into_layout();
        if self.layout.size() != layout.size() {
            return Err(TensorError::Reshape(self.layout(), layout));
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
        let TensorUntyped { device, id, .. } = self.tensor;
        let r#type = U::DATA_TYPE;
        let tensor = TensorUntyped {
            device,
            layout,
            r#type,
            id,
        };
        let phantom = PhantomData;
        Ok(Tensor { tensor, phantom })
    }
}

impl<D: Device + Clone, T: Scalar> Tensor<D, T> {
    /// Create a tensor of zeros.
    #[inline]
    pub fn zeros(device: &D, layout: impl IntoLayout) -> Self {
        let device = Arc::new(device.clone());
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

    /// Create a tensor of zeros from current device and layout.
    #[inline]
    pub fn zeros_like(&self) -> Self {
        let device = self.device().into();
        let layout = self.layout();
        let id = uid::Id::new();
        let r#type = self.r#type;
        let tensor = TensorUntyped {
            device,
            layout,
            r#type,
            id,
        };
        let phantom = PhantomData;
        Self { tensor, phantom }
    }
}
