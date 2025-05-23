use std::{
    any::{Any, TypeId},
    sync::Arc,
};

use thiserror::Error;

use super::{
    BackendOp, Device, DeviceId, OpVTable,
    allocator::{AllocOp, Allocator},
};
use crate::loom::ops::{TensorIr, TensorOp, TensorTape};

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
    id: DeviceId,
    /// Handle to a WebGPU compute device.
    device: wgpu::Device,
    /// The WebGPU command queue.
    queue: wgpu::Queue,
    /// Sends ops to execute to the backend.
    sender: flume::Sender<TensorTape>,
}

impl Device for Gpu {
    fn execute(&self, tape: TensorTape) {
        let _ = self.sender.send(tape);
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

        let id = Default::default();
        let ops = Arc::new(ops);

        let (sender, receiver) = flume::unbounded();
        let backend = {
            let device = device.clone();
            let queue = queue.clone();
            Backend { ops, device, queue }
        };
        #[cfg(not(target_arch = "wasm32"))]
        tokio::spawn(run(backend, receiver));
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(run(backend, receiver));

        let device = Gpu {
            id,
            device,
            queue,
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

async fn run(backend: Backend, receiver: flume::Receiver<TensorTape>) {
    while let Ok(tape) = receiver.recv_async().await {
        let this = tape.this;
        match execute_tape(&backend, tape) {
            Ok(_) => log::info!("{this}"),
            Err(err) => log::error!("{err}"),
        }
    }
}

fn execute_tape(backend: &Backend, tape: TensorTape) -> Result<(), Box<dyn std::error::Error>> {
    let mut allocator = Allocator::default();
    let ops = tape
        .ops
        .into_iter()
        .map(|op| allocator.alloc(op))
        .collect::<Result<Vec<_>, _>>()?;
    for op in ops {
        let io = op.io();
        backend.execute(op, io);
    }
    Ok(())
}
