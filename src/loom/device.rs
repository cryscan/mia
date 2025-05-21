use std::any::{Any, TypeId};

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::ops::TensorOp;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId;

pub trait DeviceOp<Op: TensorOp> {
    fn execute(&self, op: &Op);
}

pub trait Device {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cpu;

impl Device for Cpu {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gpu {
    /// The unique identifier of the device.
    id: uid::Id<DeviceId>,
    /// Handle to a WebGPU compute device.
    device: wgpu::Device,
    /// The WebGPU command queue.
    queue: wgpu::Queue,
}

impl Device for Gpu {}

#[derive(Debug, Clone)]
pub struct GpuBuilder {
    pub adapter: wgpu::Adapter,
    pub features: wgpu::Features,
    pub limits: wgpu::Limits,
    pub ops: FxHashMap<TypeId, fn(&Gpu, &dyn Any)>,
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
    }

    pub async fn build(&self) -> Result<Gpu, GpuBuildError> {
        let Self {
            adapter,
            features,
            limits,
            ..
        } = self.clone();

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
        let device = Gpu { id, device, queue };
        Ok(device)
    }

    pub fn limits(&mut self, limits: wgpu::Limits) -> &mut Self {
        self.limits = limits;
        self
    }

    pub fn features(&mut self, features: wgpu::Features) -> &mut Self {
        self.features = features;
        self
    }

    pub fn add_op<Op>(&mut self) -> &mut Self
    where
        Op: TensorOp,
        Gpu: DeviceOp<Op>,
    {
        let id = TypeId::of::<Op>();
        let f = |gpu: &Gpu, op: &dyn Any| match op.downcast_ref::<Op>() {
            Some(op) => <Gpu as DeviceOp<Op>>::execute(gpu, op),
            None => unreachable!(),
        };
        self.ops.insert(id, f);
        self
    }
}
