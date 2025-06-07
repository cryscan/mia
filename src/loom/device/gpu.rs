use std::{
    any::TypeId,
    cell::{RefCell, RefMut},
    sync::Arc,
};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use thiserror::Error;

use super::{
    BackData, Backend as _, Device, DeviceError, DeviceEvent, DeviceId, OpVTable,
    allocator::{AllocOp, Allocator, StashId},
};
use crate::loom::{
    ops::{BackendOp, TensorIr, TensorOp},
    platform,
    tensor::TensorId,
};

#[allow(unused)]
pub struct Backend {
    /// Handle to a WebGPU compute device.
    device: wgpu::Device,
    /// The WebGPU command queue.
    queue: wgpu::Queue,
    /// Operators that the device is able to execute.
    ops: OpVTable<Self>,
    /// Allocator that tracks buffer renaming.
    allocator: RefCell<Allocator>,
    /// Stack of GPU buffers.
    buffers: HashMap<StashId, wgpu::Buffer>,
    /// Kernel launches and uploads issued by executing a tape of ops.
    launches: Vec<Launch>,
}

impl super::Backend for Backend {
    type Data = wgpu::Buffer;

    #[inline]
    async fn execute(&mut self, op: &dyn TensorOp, io: Vec<TensorIr>) {
        let id = &op.type_id();
        match self.ops.get(id).cloned() {
            Some(f) => f(self, op, io).await,
            None => log::error!("unable to execute op of type {}", op.name()),
        }
    }

    #[inline]
    fn create(&mut self, id: TensorId, contents: &[u8]) -> Self::Data {
        use wgpu::util::DeviceExt;
        let id = self.allocator().retrieve(id);
        let data = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        self.buffers.insert(id, data.clone());
        data
    }

    #[inline]
    fn alloc(&mut self, id: TensorId, size: usize) -> Self::Data {
        let id = self.allocator().retrieve(id);
        let data = self
            .buffers
            .get(&id)
            .cloned()
            .filter(|data| data.size() as usize == size);
        let data = match data {
            Some(data) => data,
            None => self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: size as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
        };
        self.buffers.insert(id, data.clone());
        data
    }

    #[inline]
    fn fetch(&self, id: TensorId) -> Option<Self::Data> {
        let id = self.allocator().retrieve(id);
        self.buffers.get(&id).cloned()
    }
}

impl Backend {
    #[inline]
    pub fn allocator(&self) -> RefMut<Allocator> {
        self.allocator.borrow_mut()
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

        let allocator = RefCell::new(Allocator::default());
        let buffers = HashMap::default();
        let launches = Vec::new();

        let (sender, receiver) = flume::unbounded();
        let backend = Backend {
            ops,
            allocator,
            device,
            queue,
            buffers,
            launches,
        };
        platform::spawn(serve(backend, receiver));

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

    pub fn add_op<Op>(mut self) -> Self
    where
        Op: TensorOp + BackendOp<Backend>,
    {
        self.ops.insert(TypeId::of::<Op>(), |backend, op, io| {
            match op.downcast_ref::<Op>() {
                Some(op) => Box::pin(op.execute(backend, io)),
                None => unreachable!(),
            }
        });
        self
    }
}

#[derive(Debug, Clone)]
pub enum Launch {
    Kernel(Kernel),
    Upload(Upload),
}

#[derive(Debug, Clone)]
pub struct Kernel {
    pub pipeline: wgpu::ComputePipeline,
    pub bindings: Vec<wgpu::BindGroup>,
    pub dispatch: [u32; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Upload {
    pub id: TensorId,
    pub data: Arc<[u8]>,
}

async fn serve(mut backend: Backend, receiver: flume::Receiver<DeviceEvent>) {
    let mut commit = HashSet::default();

    while let Ok(event) = receiver.recv_async().await {
        match event {
            DeviceEvent::Execute { tape, sender } => {
                let id = async {
                    let ops: Vec<_> = tape
                        .ops
                        .into_iter()
                        .filter(|op| !commit.contains(&op.id()))
                        .collect();
                    for op in ops {
                        let op = backend.allocator().alloc(op)?;
                        let io = op.io();
                        let id = op.id();
                        backend.execute(&op, io).await;
                        commit.insert(id);
                    }
                    Ok(tape.id)
                }
                .await;
                _ = sender.send_async(id).await
            }
            DeviceEvent::Back { tape, sender } => {
                let data = async {
                    let ops: Vec<_> = tape
                        .ops
                        .into_iter()
                        .filter(|op| !commit.contains(&op.id()))
                        .collect();
                    for op in ops {
                        let op = backend.allocator().alloc(op)?;
                        let io = op.io();
                        let id = op.id();
                        backend.execute(&op, io).await;
                        commit.insert(id);
                    }
                    let id = tape.id;
                    backend.fetch(id).ok_or(DeviceError::Tensor(id))
                }
                .await;

                let device = backend.device.clone();
                let queue = backend.queue.clone();
                let future = async move {
                    let data = data?;
                    let (sender, receiver) = flume::bounded(0);
                    wgpu::util::DownloadBuffer::read_buffer(
                        &device,
                        &queue,
                        &data.slice(..),
                        move |data| {
                            let data = data
                                .map(|data| data.to_vec().into_boxed_slice())
                                .map(BackData)
                                .map_err(DeviceError::from);
                            _ = sender.send(data)
                        },
                    );
                    #[cfg(not(target_arch = "wasm32"))]
                    tokio::task::spawn_blocking(move || device.poll(wgpu::PollType::Wait));
                    receiver.recv_async().await?
                };
                platform::spawn(async move { _ = sender.send_async(future.await).await });
            }
            DeviceEvent::Cleanup { retain } => {
                // remove all buffers in the stash unless retained
                let f = |ir: TensorIr| backend.allocator().retrieve(ir.id);
                let f = |op: &dyn TensorOp| op.io().into_iter().map(f);
                let ids: HashSet<_> = retain
                    .iter()
                    .flat_map(|tape| tape.ops.iter().map(AsRef::as_ref).flat_map(f))
                    .collect();
                backend.buffers.retain(|id, _| ids.contains(id));

                // removes all tensors tracked in the allocator unless related to retained ones
                let f = |ir: TensorIr| ir.id;
                let f = |op: &dyn TensorOp| op.io().into_iter().map(f);
                let ids: Vec<_> = retain
                    .iter()
                    .flat_map(|tape| tape.ops.iter().map(AsRef::as_ref).flat_map(f))
                    .collect();
                backend.allocator().retain(&ids);

                // remove all committed ops unless retained
                let ids: HashSet<_> = retain
                    .into_iter()
                    .flat_map(|tape| tape.ops.into_iter().map(|op| op.id()))
                    .collect();
                commit.retain(|id| ids.contains(id));
            }
        }
    }
}
