use std::{
    any::{Any, TypeId},
    sync::Arc,
};

use super::{
    BackendOp, Device, DeviceId, OpVTable,
    allocator::{AllocOp, Allocator},
};
use crate::loom::ops::{TensorIr, TensorOp};

#[derive(Debug, Clone)]
pub struct Backend {
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
pub struct Cpu {
    /// The unique identifier of the device.
    id: uid::Id<DeviceId>,
    /// Operators that the device is able to execute.
    ops: Arc<OpVTable<Backend>>,
    /// Sends ops to execute to the backend.
    sender: flume::Sender<Box<dyn TensorOp>>,
}

impl Device for Cpu {
    fn execute(&self, op: Box<dyn TensorOp>) {
        let _ = self.sender.send(op);
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
        let id = uid::Id::new();
        let ops = Arc::new(self.ops);

        let (sender, receiver) = flume::unbounded();
        {
            let ops = ops.clone();
            let backend = Backend { ops };
            #[cfg(not(target_arch = "wasm32"))]
            tokio::spawn(run(backend, receiver));
            #[cfg(target_arch = "wasm32")]
            wasm_bindgen_futures::spawn_local(run(backend, receiver));
        }

        Cpu { id, ops, sender }
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
