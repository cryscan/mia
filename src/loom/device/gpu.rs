use std::{
    any::{Any, TypeId},
    sync::Arc,
};

use thiserror::Error;

use super::{
    BackendOp, Device, DeviceId, OpVTable,
    allocator::{AllocOp, Allocator},
};
use crate::loom::ops::{TensorIr, TensorOp};

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct Backend {
    /// Handle to a WebGPU compute device.
    pub device: wgpu::Device,
    /// The WebGPU command queue.
    pub queue: wgpu::Queue,
    /// Operators that the device is able to execute.
    ops: Arc<OpVTable<Self>>,
}

impl super::Backend for Backend {
    #[inline]
    fn execute(&self, op: Box<dyn TensorOp>, io: Vec<TensorIr>) {
        let id = &op.as_ref().type_id();
        match self.ops.get(id) {
            Some(f) => f(self, op, io),
            None => log::error!("unable to execute op of type {}", op.name()),
        }
    }
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct Gpu {
    /// The unique identifier of the device.
    id: uid::Id<DeviceId>,
    /// Handle to a WebGPU compute device.
    device: wgpu::Device,
    /// The WebGPU command queue.
    queue: wgpu::Queue,
    /// Operators that the device is able to execute.
    ops: Arc<OpVTable<Backend>>,
    /// Sends ops to execute to the backend.
    sender: flume::Sender<Box<dyn TensorOp>>,
}

impl Device for Gpu {
    fn execute(&self, op: Box<dyn TensorOp>) {
        let _ = self.sender.send(op);
    }
}

#[derive(Debug, Clone)]
pub struct GpuBuilder {
    pub adapter: wgpu::Adapter,
    pub features: wgpu::Features,
    pub limits: wgpu::Limits,
    pub ops: OpVTable<Backend>,
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
        let ops = Arc::new(ops);

        let (sender, receiver) = flume::unbounded();
        {
            let ops = ops.clone();
            let device = device.clone();
            let queue = queue.clone();
            let backend = Backend { ops, device, queue };
            #[cfg(not(target_arch = "wasm32"))]
            tokio::spawn(run(backend, receiver));
            #[cfg(target_arch = "wasm32")]
            wasm_bindgen_futures::spawn_local(run(backend, receiver));
        }

        let device = Gpu {
            id,
            device,
            queue,
            ops,
            sender,
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
        Backend: BackendOp<Op>,
    {
        let id = TypeId::of::<Op>();
        let f = |b: &Backend, op: Box<dyn TensorOp>, io| match Box::<dyn Any>::from(op).downcast() {
            Ok(op) => b.execute(*op, io),
            Err(_) => unreachable!(),
        };
        self.ops.insert(id, f);
        self
    }
}

async fn run(backend: Backend, receiver: flume::Receiver<Box<dyn TensorOp>>) {
    let mut allocator = Allocator::default();
    while let Ok(op) = receiver.recv_async().await {
        let op = match allocator.alloc(op) {
            Ok(op) => op,
            Err(err) => {
                log::error!("{}", err);
                continue;
            }
        };
        let io = op.io();
        backend.execute(op, io);
    }
}
