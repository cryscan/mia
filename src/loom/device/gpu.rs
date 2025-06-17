#[cfg(target_arch = "wasm32")]
use std::rc::Rc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;
use std::{
    any::TypeId,
    borrow::Cow,
    cell::{RefCell, RefMut},
    sync::Arc,
};

use itertools::Itertools;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use thiserror::Error;

use super::{
    BackData, Backend as _, Device, DeviceError, DeviceEvent, DeviceId, OpVTable,
    allocator::{AllocOp, Allocator, StashId},
};
use crate::loom::{
    num::Scalar,
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
    /// Kernel launches issued by executing a tape of ops.
    kernels: Vec<Kernel>,
    /// Data uploads issued by executing a tape of ops.
    uploads: Vec<Upload>,
}

impl super::Backend for Backend {
    type Data = wgpu::Buffer;

    #[inline]
    async fn execute(&mut self, op: &dyn TensorOp, io: Vec<TensorIr>) {
        let id = &op.type_id();
        match self.ops.get(id).cloned() {
            Some(f) => f(self, op, io).await,
            #[cfg(not(feature = "strict"))]
            None => log::error!("unable to execute op of type {}", op.name()),
            #[cfg(feature = "strict")]
            None => panic!("unable to execute op of type {}", op.name()),
        }
    }

    #[inline]
    fn create<'a, T, C>(&mut self, id: TensorId, contents: C) -> Self::Data
    where
        T: Scalar,
        C: Into<Cow<'a, [T]>>,
    {
        use wgpu::util::DeviceExt;
        let id = self.allocator().retrieve(id);
        let contents: Cow<_> = contents.into();
        let contents = bytemuck::cast_slice(&contents);
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
    fn alloc<T: Scalar>(&mut self, id: TensorId, len: usize) -> Self::Data {
        let id = self.allocator().retrieve(id);
        let size = size_of::<T>() * len;
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
    fn fetch(&self, id: TensorId) -> Self::Data {
        let id = self.allocator().retrieve(id);
        self.buffers
            .get(&id)
            .expect("failed to fetch buffer")
            .clone()
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
        let kernels = Vec::default();
        let uploads = Vec::default();

        let (sender, receiver) = flume::unbounded();
        let backend = Backend {
            ops,
            allocator,
            device,
            queue,
            buffers,
            kernels,
            uploads,
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

fn encode(device: &wgpu::Device, kernels: &[Kernel]) -> wgpu::CommandBuffer {
    let mut encoder = device.create_command_encoder(&Default::default());
    for kernel in kernels {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&kernel.pipeline);
        kernel
            .bindings
            .iter()
            .enumerate()
            .for_each(|(index, binding)| pass.set_bind_group(index as u32, binding, &[]));
        pass.dispatch_workgroups(kernel.dispatch[0], kernel.dispatch[1], kernel.dispatch[2]);
    }
    encoder.finish()
}

async fn serve(mut backend: Backend, receiver: flume::Receiver<DeviceEvent>) {
    let mut commit = HashSet::default();
    let mut uploads = HashMap::default();
    let mut commands = HashMap::default();

    while let Ok(event) = receiver.recv_async().await {
        match event {
            DeviceEvent::Execute { tape, sender } => {
                let id = async {
                    let ops = tape
                        .ops
                        .into_iter()
                        .filter(|op| !commit.contains(&op.id()))
                        .collect_vec();
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
                match id {
                    Ok(id) => {
                        let mut v = uploads.remove(&id).unwrap_or(Vec::new());
                        v.append(&mut backend.uploads);
                        uploads.insert(id, v);

                        #[cfg(not(target_arch = "wasm32"))]
                        let command = Arc::new(Mutex::new(None));
                        #[cfg(target_arch = "wasm32")]
                        let command = Rc::new(RefCell::new(None));

                        let mut v = commands.remove(&id).unwrap_or(Vec::new());
                        v.push(command.clone());
                        commands.insert(id, v);

                        let device = backend.device.clone();
                        let kernels = std::mem::take(&mut backend.kernels);
                        platform::dispatch(move || {
                            #[cfg(not(target_arch = "wasm32"))]
                            command
                                .lock()
                                .expect("failed to lock command")
                                .replace(encode(&device, &kernels));
                            #[cfg(target_arch = "wasm32")]
                            command.borrow_mut().replace(encode(&device, &kernels));
                            _ = sender.send(Ok(id))
                        });
                    }
                    Err(err) => _ = sender.send_async(Err(err)).await,
                }
            }
            DeviceEvent::Back { tape, sender } => {
                let id = async {
                    let ops = tape
                        .ops
                        .into_iter()
                        .filter(|op| !commit.contains(&op.id()))
                        .collect_vec();
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
                match id {
                    Ok(id) => {
                        let mut uploads = uploads.remove(&id).unwrap_or(Vec::new());
                        uploads.append(&mut backend.uploads);
                        let uploads = uploads
                            .into_iter()
                            .map(|upload| (backend.fetch(upload.id), upload.data))
                            .collect_vec();

                        let commands = commands.remove(&id).unwrap_or(Vec::new());
                        let data = backend.fetch(id);
                        let device = backend.device.clone();
                        let queue = backend.queue.clone();
                        let kernels = std::mem::take(&mut backend.kernels);
                        platform::dispatch(move || {
                            // 1. upload data into input buffers
                            uploads
                                .into_iter()
                                .for_each(|(buffer, data)| queue.write_buffer(&buffer, 0, &data));

                            // 2. encode remaining commands
                            #[cfg(not(target_arch = "wasm32"))]
                            let mut commands = commands
                                .into_iter()
                                .flat_map(|x| x.lock().expect("failed to lock command").take())
                                .collect_vec();
                            #[cfg(target_arch = "wasm32")]
                            let mut commands = commands
                                .into_iter()
                                .flat_map(|x| x.borrow_mut().take())
                                .collect_vec();
                            commands.push(encode(&device, &kernels));

                            // 3. submit and read back
                            let back = device.create_buffer(&wgpu::BufferDescriptor {
                                label: None,
                                size: data.size(),
                                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                                mapped_at_creation: false,
                            });
                            let mut encoder = device.create_command_encoder(&Default::default());
                            encoder.copy_buffer_to_buffer(&data, 0, &back, 0, data.size());
                            commands.push(encoder.finish());

                            let index = queue.submit(commands);

                            back.clone()
                                .map_async(wgpu::MapMode::Read, .., move |result| {
                                    let data = result
                                        .map(|_| back.get_mapped_range(..))
                                        .map(|data| data.to_vec().into_boxed_slice())
                                        .map(BackData)
                                        .map_err(DeviceError::from);
                                    _ = sender.send(data)
                                });

                            // 4. poll and wait
                            _ = device.poll(wgpu::PollType::WaitForSubmissionIndex(index))
                        });
                    }
                    Err(err) => _ = sender.send_async(Err(err)).await,
                }
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
                let ids = retain
                    .iter()
                    .flat_map(|tape| tape.ops.iter().map(AsRef::as_ref).flat_map(f))
                    .collect_vec();
                backend.allocator().retain(ids);

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
