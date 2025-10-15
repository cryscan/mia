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
    BackData, Backend as _, Device, DeviceError, DeviceEvent, DeviceId, ExecuteData, OpVTable,
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
    inputs: Vec<HostData>,
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
    fn try_fetch(&self, id: TensorId) -> Option<Self::Data> {
        let id = self.allocator().retrieve(id);
        self.buffers.get(&id).cloned()
    }
}

impl Backend {
    #[inline]
    pub fn allocator(&self) -> RefMut<'_, Allocator> {
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
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await?;

        let allocator = RefCell::new(Allocator::default());
        let buffers = HashMap::default();
        let kernels = Vec::default();
        let inputs = Vec::default();

        let (sender, receiver) = flume::unbounded();
        let backend = Backend {
            ops,
            allocator,
            device,
            queue,
            buffers,
            kernels,
            inputs,
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
pub struct HostData {
    pub id: TensorId,
    pub data: Arc<[u8]>,
}

/// Encodes GPU compute commands into a command buffer for execution.
///
/// # Arguments
/// * `device` - The WebGPU device used to create the command encoder.
/// * `kernels` - A slice of `Kernel` structs representing the compute operations to encode.
///
/// # Returns
/// A `wgpu::CommandBuffer` containing the encoded compute commands.
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
    let mut commands = HashMap::<_, Vec<_>>::default();
    let mut inputs = HashMap::<_, Vec<_>>::default();
    let mut outputs = HashMap::<_, Output>::default();

    #[derive(Debug, Clone)]
    enum Output {
        Ready(Result<BackData, DeviceError>),
        Future(flume::Receiver<Result<BackData, DeviceError>>),
    }

    'main: while let Ok(event) = receiver.recv_async().await {
        match event {
            DeviceEvent::Execute { tape, sender } => {
                let ops = tape
                    .ops
                    .iter()
                    .filter(|op| !commit.contains(&op.id()))
                    .cloned()
                    .collect_vec();
                commit.extend(ops.iter().map(|op| op.id()));

                for op in ops {
                    let op = backend.allocator().alloc(op).map_err(Into::into);
                    match op {
                        Ok(op) => backend.execute(&op, op.io()).await,
                        Err(err) => {
                            _ = sender.send_async(Err(err)).await;
                            continue 'main;
                        }
                    };
                }

                inputs
                    .entry(tape.id)
                    .or_default()
                    .append(&mut backend.inputs);

                let command = {
                    let (tx, rx) = flume::bounded(1);
                    commands.entry(tape.id).or_default().push(rx.clone());
                    tx
                };

                let data = backend.allocator().mermaid(&tape);
                let data = ExecuteData(data);

                let device = backend.device.clone();
                let kernels = std::mem::take(&mut backend.kernels);
                platform::dispatch(move || {
                    _ = command.send(encode(&device, &kernels));
                    _ = sender.send(Ok(data))
                });
            }
            DeviceEvent::Back { tape, sender } => {
                // check if the tape already has been executed
                if let Some(output) = outputs.get(&tape.id).cloned() {
                    match output {
                        Output::Ready(data) => _ = sender.send_async(data.clone()).await,
                        Output::Future(output) => {
                            let output = output
                                .recv_async()
                                .await
                                .map_err(DeviceError::from)
                                .flatten();
                            outputs.insert(tape.id, Output::Ready(output.clone()));
                            _ = sender.send_async(output).await;
                        }
                    }
                    continue 'main;
                }

                let ops = tape
                    .ops
                    .iter()
                    .filter(|op| !commit.contains(&op.id()))
                    .cloned()
                    .collect_vec();
                commit.extend(ops.iter().map(|op| op.id()));

                for op in ops {
                    let op = backend.allocator().alloc(op).map_err(Into::into);
                    match op {
                        Ok(op) => backend.execute(&op, op.io()).await,
                        Err(err) => {
                            _ = sender.send_async(Err(err)).await;
                            continue 'main;
                        }
                    };
                }

                let mut inputs = inputs.remove(&tape.id).unwrap_or_default();
                inputs.append(&mut backend.inputs);
                let inputs = inputs
                    .into_iter()
                    .map(|input| (backend.fetch(input.id), input.data))
                    .collect_vec();

                let output = {
                    let (tx, rx) = flume::bounded(1);
                    outputs.insert(tape.id, Output::Future(rx));
                    tx
                };

                let commands = commands.remove(&tape.id).unwrap_or_default();
                let data = backend.fetch(tape.id);
                let device = backend.device.clone();
                let queue = backend.queue.clone();
                let kernels = std::mem::take(&mut backend.kernels);
                platform::dispatch(move || {
                    // 1. upload data into input buffers
                    inputs
                        .into_iter()
                        .for_each(|(buffer, data)| queue.write_buffer(&buffer, 0, &data));

                    // 2. encode remaining commands
                    let mut commands = commands
                        .into_iter()
                        .map(|command| command.recv().expect("command encode failed"))
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
                                .map(Arc::from)
                                .map(BackData)
                                .map_err(DeviceError::from);
                            _ = sender.send(data.clone());
                            _ = output.send(data);
                        });

                    // 4. poll and wait
                    _ = device.poll(wgpu::PollType::Wait {
                        submission_index: Some(index),
                        timeout: None,
                    })
                });
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use half::f16;

    use super::*;
    use crate::loom::tensor::Tensor;

    #[tokio::test]
    async fn test_tensor_create() -> Result<(), Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::new(&Default::default());
        let Ok(adapter) = instance.request_adapter(&Default::default()).await else {
            return Ok(());
        };

        let gpu = GpuBuilder::new(adapter).add_default_ops().build().await?;

        const C: usize = 1024;
        const T: usize = 768;

        let data: Arc<[f16]> = (0..C * T).map(|x| f16::from_f32(x as f32)).collect();

        let tensor = Tensor::create(gpu.clone(), [C, T], data.clone())?;
        let mermaid = tensor.clone().mermaid();
        let x = tensor.clone().back();
        let y = tensor.back();

        let (mermaid, x, y) = tokio::try_join!(mermaid, x, y)?;

        println!("{mermaid}");

        let r#ref = data.into_iter().copied().collect::<Box<[_]>>();
        assert_eq!(x, r#ref);
        assert_eq!(y, r#ref);

        Ok(())
    }
}
