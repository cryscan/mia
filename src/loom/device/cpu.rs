use std::{
    any::TypeId,
    cell::{RefCell, RefMut},
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock},
};

use itertools::Itertools;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use super::{
    BackData, Backend as _, Device, DeviceEvent, DeviceId, OpVTable,
    allocator::{AllocOp, Allocator, StashId},
};
use crate::loom::{
    num::Scalar,
    ops::{BackendOp, TensorIr, TensorOp},
    platform,
    tensor::TensorId,
};

#[derive(Debug, Clone)]
pub struct Buffer(Arc<RwLock<Box<[u8]>>>);

impl From<Box<[u8]>> for Buffer {
    fn from(value: Box<[u8]>) -> Self {
        Self(Arc::new(RwLock::new(value)))
    }
}

impl From<Vec<u8>> for Buffer {
    fn from(value: Vec<u8>) -> Self {
        value.into_boxed_slice().into()
    }
}

impl Buffer {
    #[inline]
    pub fn read(&self) -> impl Deref<Target = Box<[u8]>> {
        self.0.read().expect("failed to lock buffer")
    }

    #[inline]
    pub fn write(&self) -> impl DerefMut<Target = Box<[u8]>> {
        self.0.write().expect("failed to lock buffer")
    }

    #[inline]
    pub fn read_slice<T: Scalar>(&self) -> impl Deref<Target = [T]> {
        let lock = self.read();
        let phantom = PhantomData;
        BufferReader { lock, phantom }
    }

    #[inline]
    pub fn write_slice<T: Scalar>(&self) -> impl DerefMut<Target = [T]> {
        let lock = self.write();
        let phantom = PhantomData;
        BufferWriter { lock, phantom }
    }

    #[inline]
    pub fn into_inner(self) -> Box<[u8]> {
        match Arc::try_unwrap(self.0) {
            Ok(inner) => inner.into_inner().expect("lock poisoned"),
            Err(this) => Self(this).read_slice().to_vec().into_boxed_slice(),
        }
    }
}

pub struct BufferReader<R, T> {
    lock: R,
    phantom: PhantomData<T>,
}

impl<R, T> std::ops::Deref for BufferReader<R, T>
where
    R: Deref<Target = Box<[u8]>>,
    T: Scalar,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        bytemuck::cast_slice(&self.lock)
    }
}

pub struct BufferWriter<W, T> {
    lock: W,
    phantom: PhantomData<T>,
}

impl<R, T> std::ops::Deref for BufferWriter<R, T>
where
    R: Deref<Target = Box<[u8]>>,
    T: Scalar,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        bytemuck::cast_slice(&self.lock)
    }
}

impl<R, T> std::ops::DerefMut for BufferWriter<R, T>
where
    R: DerefMut<Target = Box<[u8]>>,
    T: Scalar,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        bytemuck::cast_slice_mut(&mut self.lock)
    }
}

#[allow(unused)]
pub struct Backend {
    /// Operators that the device is able to execute.
    ops: OpVTable<Self>,
    /// Allocator that tracks buffer renaming.
    allocator: RefCell<Allocator>,
    /// Stack of CPU buffers.
    buffers: HashMap<StashId, Buffer>,
}

impl super::Backend for Backend {
    type Data = Buffer;

    #[inline]
    async fn execute(&mut self, op: &dyn TensorOp, io: Vec<TensorIr>) {
        let id = &op.type_id();
        match self.ops.get(id) {
            Some(f) => f(self, op, io).await,
            #[cfg(feature = "strict")]
            None => log::error!("unable to execute op of type {}", op.name()),
            #[cfg(not(feature = "strict"))]
            None => panic!("unable to execute op of type {}", op.name()),
        }
    }

    #[inline]
    fn create(&mut self, id: TensorId, contents: &[u8]) -> Self::Data {
        let id = self.allocator().retrieve(id);
        let data = Self::Data::from(contents.to_vec());
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
            .filter(|data| data.read_slice::<u8>().len() == size);
        let data = match data {
            Some(data) => data,
            None => vec![0; size].into(),
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
pub struct Cpu {
    /// The unique identifier of the device.
    id: DeviceId,
    /// Sends ops to execute to the backend.
    sender: flume::Sender<DeviceEvent>,
}

impl Device for Cpu {
    fn execute(&self, event: DeviceEvent) {
        _ = self.sender.send(event)
    }
}

#[derive(Default)]
pub struct CpuBuilder {
    pub ops: OpVTable<Backend>,
}

impl CpuBuilder {
    pub fn new() -> Self {
        Self::default().add_op::<AllocOp>()
    }

    pub async fn build(self) -> Cpu {
        let ops = self.ops;
        let allocator = RefCell::new(Allocator::default());
        let buffers = HashMap::default();

        let (sender, receiver) = flume::unbounded();
        let backend = Backend {
            ops,
            allocator,
            buffers,
        };
        platform::spawn(serve(backend, receiver));

        let id = Default::default();
        Cpu { id, sender }
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

async fn serve(mut backend: Backend, receiver: flume::Receiver<DeviceEvent>) {
    let mut commit = HashSet::default();

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
                _ = sender.send_async(id).await
            }
            DeviceEvent::Back { tape, sender } => {
                let data = async {
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
                    let id = tape.id;
                    Ok(backend.fetch(id))
                }
                .await
                .map(Buffer::into_inner)
                .map(BackData);
                _ = sender.send_async(data).await
            }
            DeviceEvent::Cleanup { retain } => {
                // remove all buffers in the stack unless retained
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
                    .iter()
                    .flat_map(|tape| tape.ops.iter().map(|op| op.id()))
                    .collect();
                commit.retain(|id| ids.contains(id));
            }
        }
    }
}
