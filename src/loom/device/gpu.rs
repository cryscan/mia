use std::{
    any::TypeId,
    sync::{Arc, RwLock},
};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use thiserror::Error;

use super::{
    BackData, Backend as _, Device, DeviceError, DeviceEvent, DeviceId, OpVTable,
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
    buffers: Arc<RwLock<HashMap<TensorId, wgpu::Buffer>>>,
}

impl super::Backend for Backend {
    type Buffer = wgpu::Buffer;

    #[inline]
    fn execute(&self, op: &dyn TensorOp, io: Vec<TensorIr>) {
        let id = &op.type_id();
        match self.ops.get(id) {
            Some(f) => f(self, op, io),
            None => log::error!("unable to execute op of type {}", op.name()),
        }
    }

    #[inline]
    fn alloc(&self, id: TensorId, contents: &[u8]) -> Self::Buffer {
        use wgpu::util::DeviceExt;

        let label = format!("tensor: {id}");
        let label = Some(label.as_str());
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents,
                usage,
            });
        self.buffers
            .write()
            .expect("failed to lock")
            .insert(id, buffer.clone());
        buffer
    }

    #[inline]
    fn fetch(&self, id: TensorId) -> Option<Self::Buffer> {
        self.buffers
            .read()
            .expect("failed to lock")
            .get(&id)
            .cloned()
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

    /// Reads a tensor of `id` back.
    async fn read(&self, id: TensorId) -> Result<Box<[u8]>, DeviceError> {
        let (sender, receiver) = flume::bounded(0);
        let buffer = self.fetch(id).ok_or(DeviceError::Tensor(id))?;

        wgpu::util::DownloadBuffer::read_buffer(
            &self.device,
            &self.queue,
            &buffer.slice(..),
            move |buffer| {
                let data = match buffer {
                    Ok(buffer) => Ok(buffer.to_vec().into_boxed_slice()),
                    Err(err) => Err(DeviceError::from(err)),
                };
                _ = sender.send(data)
            },
        );

        #[cfg(not(target_arch = "wasm32"))]
        {
            let device = self.device.clone();
            tokio::task::spawn_blocking(move || device.poll(wgpu::PollType::Wait));
        }

        receiver.recv_async().await?
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
        let buffers = Arc::new(RwLock::new(HashMap::default()));

        let (sender, receiver) = flume::unbounded();
        let backend = Backend {
            ops,
            device,
            queue,
            buffers,
        };
        super::spawn(serve(backend, receiver));

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
    let mut commit = HashSet::default();

    let execute = |commit: &mut HashSet<_>, tape: TensorTape| {
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
    };

    'main: while let Ok(event) = receiver.recv_async().await {
        match event {
            DeviceEvent::Execute { tape, sender } => {
                let id = execute(&mut commit, tape);
                _ = sender.send_async(id).await
            }
            DeviceEvent::Back { tape, sender } => {
                let id = match execute(&mut commit, tape) {
                    Ok(id) => id,
                    Err(err) => {
                        _ = sender.send_async(Err(err)).await;
                        continue 'main;
                    }
                };
                let backend = backend.clone();
                super::spawn(async move {
                    let data = backend
                        .read(id)
                        .await
                        .map(Into::into)
                        .map(|data| BackData { data, id });
                    _ = sender.send_async(data).await
                });
            }
            DeviceEvent::Cleanup { retain } => {
                let retain: HashSet<_> = retain
                    .into_iter()
                    .flat_map(|tape| tape.ops.into_iter().map(|op| op.id()))
                    .collect();
                commit.retain(|id| retain.contains(id));
            }
        }
    }
}
