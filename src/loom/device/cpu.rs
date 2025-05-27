use std::{any::TypeId, sync::Arc};

use rustc_hash::FxHashSet as HashSet;

use super::{
    Backend as _, Device, DeviceError, DeviceEvent, DeviceId, OpVTable,
    allocator::{AllocOp, Allocator},
};
use crate::loom::{
    ops::{BackendOp, TensorIr, TensorOp, TensorOpId, TensorTape},
    tensor::TensorId,
};

#[derive(Debug, Clone)]
pub struct Backend {
    /// Operators that the device is able to execute.
    ops: Arc<OpVTable<Self>>,
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
        let _ = self.sender.send(event);
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
        let id = Default::default();
        let ops = Arc::new(self.ops);

        let (sender, receiver) = flume::unbounded();
        let backend = Backend { ops };
        #[cfg(not(target_arch = "wasm32"))]
        tokio::spawn(run(backend, receiver));
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(run(backend, receiver));

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

async fn run(backend: Backend, receiver: flume::Receiver<DeviceEvent>) {
    let mut commit = HashSet::default();
    while let Ok(event) = receiver.recv_async().await {
        match event {
            DeviceEvent::Execute { tape, sender } => {
                let result = execute(&backend, &mut commit, tape);
                let _ = sender.send(result);
            }
            DeviceEvent::ExecuteBack { .. } => todo!(),
        }
    }
}

fn execute(
    backend: &Backend,
    commit: &mut HashSet<TensorOpId>,
    tape: TensorTape,
) -> Result<TensorId, DeviceError> {
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
