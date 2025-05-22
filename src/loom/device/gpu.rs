use std::{
    any::{Any, TypeId},
    sync::Arc,
};

use thiserror::Error;

use super::{Device, DeviceId, DeviceOp, OpVTable, allocator::AllocOp};
use crate::loom::ops::{TensorIr, TensorOp};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gpu {
    /// The unique identifier of the device.
    pub id: uid::Id<DeviceId>,
    /// Handle to a WebGPU compute device.
    pub device: wgpu::Device,
    /// The WebGPU command queue.
    pub queue: wgpu::Queue,
    /// Operators that the device is able to execute.
    pub ops: Arc<OpVTable<Self>>,
}

impl Device for Gpu {
    #[inline]
    fn execute_dyn(&self, op: Box<dyn TensorOp>, io: Vec<TensorIr>) {
        assert_eq!(op.io().len(), io.len());
        match self.ops.get(&op.as_ref().type_id()) {
            Some(f) => f(self, op, io),
            None => log::error!("unable to execute op of type {}", op.name()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuBuilder {
    pub adapter: wgpu::Adapter,
    pub features: wgpu::Features,
    pub limits: wgpu::Limits,
    pub ops: OpVTable<Gpu>,
}

#[derive(Debug, Error)]
pub enum GpuBuildError {
    #[error("failed to request adaptor")]
    RequestAdapterError(#[from] wgpu::RequestAdapterError),
    #[error("failed to request device")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
}

impl GpuBuilder {
    pub fn new(adapter: wgpu::Adapter) -> Self {
        let features = wgpu::Features::empty();
        #[cfg(feature = "subgroup-ops")]
        let features = features | wgpu::Features::SUBGROUP;
        Self {
            adapter,
            features,
            limits: Default::default(),
            ops: Default::default(),
        }
        .add_op::<AllocOp>()
    }

    pub async fn build(self) -> Result<Gpu, GpuBuildError> {
        let Self {
            adapter,
            features,
            limits,
            ops,
        } = self;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await?;

        let id = uid::Id::new();
        let ops = ops.into();
        let device = Gpu {
            id,
            device,
            queue,
            ops,
        };
        Ok(device)
    }

    pub fn limits(mut self, limits: wgpu::Limits) -> Self {
        self.limits = limits;
        self
    }

    pub fn features(mut self, features: wgpu::Features) -> Self {
        self.features = features;
        self
    }

    pub fn add_op<Op>(mut self) -> Self
    where
        Op: TensorOp,
        Gpu: DeviceOp<Op>,
    {
        let id = TypeId::of::<Op>();
        let f = |gpu: &Gpu, op: Box<dyn TensorOp>, io| match Box::<dyn Any>::from(op).downcast() {
            Ok(op) => gpu.execute(*op, io),
            Err(_) => unreachable!(),
        };
        self.ops.insert(id, f);
        self
    }
}
