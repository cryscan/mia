use std::{
    any::TypeId,
    sync::{Arc, RwLock},
};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

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
    /// Operators that the device is able to execute.
    ops: Arc<OpVTable<Self>>,
    /// Pool of CPU buffers.
    buffers: Arc<RwLock<HashMap<TensorId, Arc<[u8]>>>>,
}

impl super::Backend for Backend {
    type Buffer = Arc<[u8]>;

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
        let buffer: Self::Buffer = contents.to_vec().into();
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
        let ops = Arc::new(self.ops);
        let buffers = Arc::new(RwLock::new(HashMap::default()));

        let (sender, receiver) = flume::unbounded();
        let backend = Backend { ops, buffers };
        super::spawn(serve(backend, receiver));

        let id = Default::default();
        Cpu { id, sender }
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
                let data = backend
                    .fetch(id)
                    .map(|data| BackData { id, data })
                    .ok_or(DeviceError::Tensor(id));
                _ = sender.send_async(data).await
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
