use std::any::{Any, TypeId};

use rustc_hash::FxHashMap as HashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::ops::TensorOp;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId;

/// Implemented for each [`Device`] for each [`TensorOp`].
/// Defines an `op`'s actual execution on the device.
pub trait DeviceOp<Op: TensorOp> {
    fn execute(&self, op: Op);
}

pub trait Device {
    fn execute_dyn(&self, op: Box<dyn TensorOp>);
}

type OpVTable<D> = HashMap<TypeId, fn(&D, Box<dyn TensorOp>)>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cpu {
    /// The unique identifier of the device.
    id: uid::Id<DeviceId>,
    /// Operators that the device is able to execute.
    ops: OpVTable<Self>,
}

impl Device for Cpu {
    #[inline]
    fn execute_dyn(&self, op: Box<dyn TensorOp>) {
        let id = op.as_ref().type_id();
        match self.ops.get(&id) {
            Some(f) => f(self, op),
            None => log::error!("unable to execute op of type {id:?}"),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct CpuBuilder {
    pub ops: OpVTable<Cpu>,
}

impl CpuBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(&self) -> Cpu {
        let id = uid::Id::new();
        let ops = self.ops.clone();
        Cpu { id, ops }
    }

    pub fn add_op<Op>(&mut self) -> &mut Self
    where
        Op: TensorOp,
        Cpu: DeviceOp<Op>,
    {
        let id = TypeId::of::<Op>();
        let f = |cpu: &Cpu, op: Box<dyn TensorOp>| match Box::<dyn Any>::from(op).downcast() {
            Ok(op) => cpu.execute(*op),
            Err(_) => unreachable!(),
        };
        self.ops.insert(id, f);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gpu {
    /// The unique identifier of the device.
    id: uid::Id<DeviceId>,
    /// Handle to a WebGPU compute device.
    device: wgpu::Device,
    /// The WebGPU command queue.
    queue: wgpu::Queue,
    /// Operators that the device is able to execute.
    ops: OpVTable<Self>,
}

impl Device for Gpu {
    #[inline]
    fn execute_dyn(&self, op: Box<dyn TensorOp>) {
        let id = op.as_ref().type_id();
        match self.ops.get(&id) {
            Some(f) => f(self, op),
            None => log::error!("unable to execute op of type {id:?}"),
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
    }

    pub async fn build(&self) -> Result<Gpu, GpuBuildError> {
        let Self {
            adapter,
            features,
            limits,
            ops,
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
        let device = Gpu {
            id,
            device,
            queue,
            ops,
        };
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
        let f = |gpu: &Gpu, op: Box<dyn TensorOp>| match Box::<dyn Any>::from(op).downcast() {
            Ok(op) => gpu.execute(*op),
            Err(_) => unreachable!(),
        };
        self.ops.insert(id, f);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{Cpu, CpuBuilder, Device, DeviceOp};
    use crate::loom::ops::{TensorIr, TensorOp};

    #[test]
    fn test_add_op() {
        struct PhonyOp<const N: usize>;

        impl<const N: usize> TensorOp for PhonyOp<N> {
            fn io(&self) -> Vec<TensorIr> {
                vec![]
            }
        }

        impl<const N: usize> DeviceOp<PhonyOp<N>> for Cpu {
            fn execute(&self, _op: PhonyOp<N>) {
                println!("execute phony op: {N}");
            }
        }

        let cpu = CpuBuilder::new()
            .add_op::<PhonyOp<0>>()
            .add_op::<PhonyOp<1>>()
            .add_op::<PhonyOp<2>>()
            .add_op::<PhonyOp<3>>()
            .build();
        let ops: Vec<Box<dyn TensorOp>> = vec![
            Box::new(PhonyOp::<3>),
            Box::new(PhonyOp::<2>),
            Box::new(PhonyOp::<1>),
            Box::new(PhonyOp::<0>),
        ];
        ops.into_iter().for_each(|op| cpu.execute_dyn(op));
    }
}
