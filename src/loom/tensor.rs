use std::{marker::PhantomData, sync::Arc};

use casey::snake;
use derive_more::{Deref, DerefMut, Display, From, Into};
use itertools::Itertools;
use thiserror::Error;

use super::{
    device::{Cpu, Device, Gpu},
    layout::Layout,
};
use crate::{future::Future, num::Scalar};

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("tensor creation error: size not match, layout {0}'s size not match data len {1}")]
    Create(Layout, usize),
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId;

#[derive(Debug, Clone)]
pub struct Tensor<D: Device, T: Scalar> {
    pub device: D,
    pub layout: Layout,
    pub slice: Slice,
    pub data: Arc<D::Data>,
    pub id: uid::Id<TensorId>,
    pub phantom: PhantomData<T>,
}

impl<D: Device, T: Scalar> Drop for Tensor<D, T> {
    fn drop(&mut self) {
        match Arc::strong_count(&self.data) {
            0 | 1 => self.device.dealloc(self.data.clone()),
            _ => (),
        }
    }
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

pub trait TensorInit<D: Device, T: Scalar>: Sized {
    /// Init a tensor of zeros.
    fn zeros(device: D, layout: Layout) -> impl Future<Self>;
    /// Create a tensor from data.
    fn create(device: D, layout: Layout, data: &[T]) -> impl Future<Result<Self, TensorError>>;
}

impl<T: Scalar> TensorInit<Cpu, T> for Tensor<Cpu, T> {
    async fn zeros(device: Cpu, layout: Layout) -> Self {
        let data = device.alloc::<T>(layout.size(), ()).await;
        let slice = Slice(vec![Axis::Full; layout.len()]);
        let id = uid::Id::new();
        let phantom = PhantomData;
        Self {
            device,
            layout,
            slice,
            data,
            id,
            phantom,
        }
    }

    async fn create(device: Cpu, layout: Layout, data: &[T]) -> Result<Self, TensorError> {
        if layout.size() != data.len() {
            return Err(TensorError::Create(layout, data.len()));
        }
        let data = device.create(data, ()).await;
        let slice = Slice(vec![Axis::Full; layout.len()]);
        let id = uid::Id::new();
        let phantom = PhantomData;
        Ok(Self {
            device,
            layout,
            slice,
            data,
            id,
            phantom,
        })
    }
}

impl<T: Scalar> TensorInit<Gpu, T> for Tensor<Gpu, T> {
    async fn zeros(device: Gpu, layout: Layout) -> Self {
        let data = device.alloc::<T>(layout.size(), Self::PARAMS).await;
        let slice = Slice(vec![Axis::Full; layout.len()]);
        let id = uid::Id::new();
        let phantom = PhantomData;
        Self {
            device,
            layout,
            slice,
            data,
            id,
            phantom,
        }
    }

    async fn create(device: Gpu, layout: Layout, data: &[T]) -> Result<Self, TensorError> {
        if layout.size() != data.len() {
            return Err(TensorError::Create(layout, data.len()));
        }
        let data = device.create(data, Self::PARAMS).await;
        let slice = Slice(vec![Axis::Full; layout.len()]);
        let id = uid::Id::new();
        let phantom = PhantomData;
        Ok(Self {
            device,
            layout,
            slice,
            data,
            id,
            phantom,
        })
    }
}

pub trait TensorTo<D: Device, T: Scalar> {
    /// Send a tensor to another device.
    fn to(self, device: D) -> impl Future<Tensor<D, T>>;
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
        let slice = self.slice.clone();
        let id = uid::Id::new();
        let phantom = PhantomData;
        Tensor {
            device,
            layout,
            data,
            slice,
            id,
            phantom,
        }
    }
}

impl<T: Scalar> TensorTo<Cpu, T> for Tensor<Gpu, T> {
    #[inline]
    async fn to(self, device: Cpu) -> Tensor<Cpu, T> {
        let data = self.device.read(&self.data).await.into();
        let layout = self.layout.clone();
        let slice = self.slice.clone();
        let id = uid::Id::new();
        let phantom = PhantomData;
        Tensor {
            device,
            layout,
            data,
            slice,
            id,
            phantom,
        }
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
pub struct Slice(pub Vec<Axis>);

macro_rules! impl_slice_from {
    ($t:ident) => {
        impl<$t: Into<Axis>> From<$t> for Slice {
            #[inline]
            fn from(snake!($t): $t) -> Self {
                Self(vec![snake!($t).into()])
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
                Self(vec![$(snake!($t).into()),+])
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

#[cfg(test)]
mod tests {
    use super::Slice;

    #[test]
    fn test_slice() {
        println!("{}", Slice::from(1));
        println!("{}", Slice::from(..));
        println!("{}", Slice::from((0, 1)));
        println!("{}", Slice::from((1, ..)));
        println!("{}", Slice::from((0, .., 1, 5)));
    }
}
