use std::{
    any::TypeId,
    cell::{RefCell, RefMut},
    sync::Arc,
};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use super::{
    BackData, Backend as _, Device, DeviceError, DeviceEvent, DeviceId, OpVTable,
    allocator::{AllocOp, Allocator, StackId},
};
use crate::loom::{
    ops::{BackendOp, TensorIr, TensorOp},
    tensor::TensorId,
};

#[allow(unused)]
#[derive(Debug)]
pub struct Backend {
    /// Operators that the device is able to execute.
    ops: OpVTable<Self>,
    /// Allocator that tracks buffer renaming.
    allocator: RefCell<Allocator>,
    /// Stack of CPU buffers.
    buffers: HashMap<StackId, Arc<[u8]>>,
}

impl super::Backend for Backend {
    type Data = Arc<[u8]>;

    #[inline]
    fn execute(&mut self, op: &dyn TensorOp, io: Vec<TensorIr>) {
        let id = &op.type_id();
        match self.ops.get(id) {
            Some(f) => f(self, op, io),
            None => log::error!("unable to execute op of type {}", op.name()),
        }
    }

    #[inline]
    fn create(&mut self, id: TensorId, contents: &[u8]) -> Self::Data {
        let id = self.allocator().retrieve(id);
        let data: Self::Data = contents.to_vec().into();
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
            .filter(|data| data.len() == size);
        let data = match data {
            Some(data) => data,
            None => vec![0; size].into(),
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

#[derive(Debug, Default, Clone)]
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
        super::spawn(serve(backend, receiver));

        let id = Default::default();
        Cpu { id, sender }
    }

    pub fn add_op<Op: TensorOp + BackendOp<Backend>>(mut self) -> Self {
        let id = TypeId::of::<Op>();
        let f = |backend: &mut Backend, op: &dyn TensorOp, io| match op.downcast_ref::<Op>() {
            Some(op) => op.execute(backend, io),
            None => unreachable!(),
        };
        self.ops.insert(id, f);
        self
    }
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
                        backend.execute(&op, io);
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
                        backend.execute(&op, io);
                        commit.insert(id);
                    }
                    let id = tape.id;
                    backend.fetch(id).ok_or(DeviceError::Tensor(id))
                }
                .await
                .map(BackData);
                _ = sender.send_async(data).await
            }
            DeviceEvent::Cleanup { retain } => {
                let retain: HashSet<_> = retain
                    .into_iter()
                    .flat_map(|tape| tape.ops.into_iter().map(|op| op.id()))
                    .collect();
                commit.retain(|id| retain.contains(id));
                // std::mem::take(&mut backend.allocator);
                // std::mem::take(&mut backend.buffers);
            }
        }
    }
}
