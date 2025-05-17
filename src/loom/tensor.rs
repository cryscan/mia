use std::{
    fmt::{Debug, Formatter},
    marker::PhantomData,
    sync::Arc,
};

use derive_more::{Deref, DerefMut};
use thiserror::Error;

use super::{
    device::{Cpu, Device, Gpu},
    layout::Layout,
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

/// [`TensorUntyped`] is not really untyped, but a tensor that is not statically typed.
pub struct TensorUntyped<D: Device> {
    device: D,
    layout: Layout,
    data: Arc<D::Data>,
    id: uid::Id<TensorId>,
    r#type: DataType,
}

impl<D: Device + Debug> Debug for TensorUntyped<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorUntyped")
            .field("device", &self.device)
            .field("layout", &self.layout)
            .field("id", &self.id)
            .field("type", &self.r#type)
            .finish()
    }
}

impl<D: Device> PartialEq for TensorUntyped<D> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<D: Device> Eq for TensorUntyped<D> {}

impl<D: Device> Drop for TensorUntyped<D> {
    fn drop(&mut self) {
        match Arc::strong_count(&self.data) {
            0 | 1 => self.device.dealloc(self.data.clone()),
            _ => (),
        }
    }
}

impl<D: Device + Clone> Clone for TensorUntyped<D> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            layout: self.layout.clone(),
            data: self.data.clone(),
            id: self.id,
            r#type: self.r#type,
        }
    }
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
}

/// A statically typed tensor. Fits into typed APIs.
#[derive(Debug, Clone, PartialEq, Eq, Deref, DerefMut)]
pub struct Tensor<D: Device, T: Scalar> {
    #[deref]
    #[deref_mut]
    tensor: TensorUntyped<D>,
    phantom: PhantomData<T>,
}

impl<T: Scalar> Tensor<Cpu, T> {
    #[inline]
    pub fn data(&self) -> &[T] {
        bytemuck::cast_slice(&self.data)
    }
}

impl<T: Scalar> Tensor<Gpu, T> {
    pub const PARAMS: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
        .union(wgpu::BufferUsages::COPY_DST)
        .union(wgpu::BufferUsages::COPY_SRC);
}

#[cfg_attr(not(target_arch = "wasm32"), trait_variant::make(Send))]
pub trait TensorInit<D: Device, T: Scalar>: Sized {
    /// Init a tensor of zeros.
    async fn zeros(device: D, layout: Layout) -> Self;
    /// Create a tensor from data.
    async fn create(device: D, layout: Layout, data: &[T]) -> Result<Self, TensorError>;
}

impl<T: Scalar> TensorInit<Cpu, T> for Tensor<Cpu, T> {
    async fn zeros(device: Cpu, layout: Layout) -> Self {
        let data = device.alloc::<T>(layout.size(), ()).await;
        let id = uid::Id::new();
        let r#type = T::DATA_TYPE;
        let phantom = PhantomData;
        let tensor = TensorUntyped {
            device,
            layout,
            data,
            id,
            r#type,
        };
        Self { tensor, phantom }
    }

    async fn create(device: Cpu, layout: Layout, data: &[T]) -> Result<Self, TensorError> {
        if layout.size() != data.len() {
            return Err(TensorError::Create(layout, data.len()));
        }
        let data = device.create(data, ()).await;
        let id = uid::Id::new();
        let r#type = T::DATA_TYPE;
        let phantom = PhantomData;
        let tensor = TensorUntyped {
            device,
            layout,
            data,
            id,
            r#type,
        };
        Ok(Self { tensor, phantom })
    }
}

impl<T: Scalar> TensorInit<Gpu, T> for Tensor<Gpu, T> {
    async fn zeros(device: Gpu, layout: Layout) -> Self {
        let data = device.alloc::<T>(layout.size(), Self::PARAMS).await;
        let id = uid::Id::new();
        let r#type = T::DATA_TYPE;
        let phantom = PhantomData;
        let tensor = TensorUntyped {
            device,
            layout,
            data,
            id,
            r#type,
        };
        Self { tensor, phantom }
    }

    async fn create(device: Gpu, layout: Layout, data: &[T]) -> Result<Self, TensorError> {
        if layout.size() != data.len() {
            return Err(TensorError::Create(layout, data.len()));
        }
        let data = device.create(data, Self::PARAMS).await;
        let id = uid::Id::new();
        let r#type = T::DATA_TYPE;
        let phantom = PhantomData;
        let tensor = TensorUntyped {
            device,
            layout,
            data,
            id,
            r#type,
        };
        Ok(Self { tensor, phantom })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), trait_variant::make(Send))]
pub trait TensorTo<D: Device, T: Scalar> {
    /// Send a tensor to another device.
    async fn to(self, device: D) -> Tensor<D, T>;
}

impl<T: Scalar> TensorTo<Cpu, T> for Tensor<Cpu, T> {
    #[inline]
    async fn to(self, _device: Cpu) -> Tensor<Cpu, T> {
        self
    }
}

impl<T: Scalar> TensorTo<Gpu, T> for Tensor<Cpu, T> {
    #[inline]
    async fn to(self, device: Gpu) -> Tensor<Gpu, T> {
        let data = device.create(self.data(), Tensor::<Gpu, T>::PARAMS).await;
        let layout = self.layout.clone();
        let id = uid::Id::new();
        let r#type = T::DATA_TYPE;
        let phantom = PhantomData;
        let tensor = TensorUntyped {
            device,
            layout,
            data,
            id,
            r#type,
        };
        Tensor { tensor, phantom }
    }
}

impl<T: Scalar> TensorTo<Cpu, T> for Tensor<Gpu, T> {
    #[inline]
    async fn to(self, device: Cpu) -> Tensor<Cpu, T> {
        let data = self.device.read(&self.data).await.into();
        let layout = self.layout.clone();
        let id = uid::Id::new();
        let r#type = T::DATA_TYPE;
        let phantom = PhantomData;
        let tensor = TensorUntyped {
            device,
            layout,
            data,
            id,
            r#type,
        };
        Tensor { tensor, phantom }
    }
}

impl<T: Scalar> TensorTo<Gpu, T> for Tensor<Gpu, T> {
    #[inline]
    async fn to(self, device: Gpu) -> Tensor<Gpu, T> {
        if self.device == device {
            return self;
        }
        let cpu: Tensor<Cpu, T> = self.to(Cpu::new()).await;
        cpu.to(device).await
    }
}

impl<D: Device, T: Scalar> Tensor<D, T> {
    /// Transform the tensor into an untyped one.
    #[inline]
    pub fn into_untyped(self) -> TensorUntyped<D> {
        self.tensor
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
