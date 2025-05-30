use std::{
    any::TypeId,
    sync::{Arc, Mutex},
};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use thiserror::Error;

use super::{
    Backend as _, Device, DeviceEvent, DeviceId, OpVTable,
    allocator::{AllocOp, Allocator},
};
use crate::loom::{
    ops::{BackendOp, TensorIr, TensorOp, TensorTape},
    tensor::TensorId,
};

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct Backend {
    /// Handle to a WebGPU compute device.
    device: wgpu::Device,
    /// The WebGPU command queue.
    queue: wgpu::Queue,
    /// Operators that the device is able to execute.
    ops: Arc<OpVTable<Self>>,
    /// Pool of GPU buffers.
    buffers: Arc<Mutex<HashMap<TensorId, wgpu::Buffer>>>,
}

impl super::Backend for Backend {
    #[inline]
    fn execute(&self, op: &dyn TensorOp, io: Vec<TensorIr>) {
        let id = &op.type_id();
        match self.ops.get(id) {
            Some(f) => f(self, op, io),
            None => log::error!("unable to execute op of type {}", op.name()),
        }
    }
}

impl Backend {
    #[inline]
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    #[inline]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct Gpu {
    /// The unique identifier of the device.
    id: DeviceId,
    /// Sends ops to execute to the backend.
    sender: flume::Sender<DeviceEvent>,
}

impl Device for Gpu {
    fn execute(&self, event: DeviceEvent) {
        _ = self.sender.send(event)
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

        let ops = Arc::new(ops);
        let buffers = Arc::new(Mutex::new(HashMap::default()));

        let (sender, receiver) = flume::unbounded();
        let backend = Backend {
            ops,
            device,
            queue,
            buffers,
        };
        #[cfg(not(target_arch = "wasm32"))]
        tokio::spawn(serve(backend, receiver));
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(serve(backend, receiver));

        let id = Default::default();
        let device = Gpu { id, sender };
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

    pub fn add_op<Op: TensorOp + BackendOp<Backend>>(mut self) -> Self {
        let id = TypeId::of::<Op>();
        let f = |backend: &Backend, op: &dyn TensorOp, io| match op.downcast_ref::<Op>() {
            Some(op) => op.execute(backend, io),
            None => unreachable!(),
        };
        self.ops.insert(id, f);
        self
    }
}

async fn serve(backend: Backend, receiver: flume::Receiver<DeviceEvent>) {
    let commit = Arc::new(Mutex::new(HashSet::default()));

    let execute = {
        let backend = backend.clone();
        let commit = commit.clone();
        move |tape: TensorTape| {
            let mut commit = commit.lock().unwrap();
            let mut allocator = Allocator::default();
            let ops: Vec<_> = tape
                .ops
                .into_iter()
                .filter(|op| !commit.contains(&op.id()))
                .collect();
            for op in ops {
                let op = allocator.alloc(op)?;
                let io = op.io();
                let id = op.id();
                backend.execute(&op, io);
                commit.insert(id);
            }
            Ok(tape.id)
        }
    };

    while let Ok(event) = receiver.recv_async().await {
        match event {
            DeviceEvent::Execute { tape, sender } => _ = sender.send_async(execute(tape)).await,
            DeviceEvent::Back { .. } => todo!(),
            DeviceEvent::Cleanup { retain } => {
                let retain: HashSet<_> = retain
                    .into_iter()
                    .flat_map(|tape| tape.ops.into_iter().map(|op| op.id()))
                    .collect();
                let mut commit = commit.lock().unwrap();
                commit.retain(|id| retain.contains(id));
            }
        }
    }
}
