use std::{
    any::{Any, TypeId},
    sync::Arc,
};

use super::{
    BackendOp, Device, DeviceEvent, DeviceId, OpVTable,
    allocator::{AllocOp, Allocator},
};
use crate::loom::ops::{TensorIr, TensorOp, TensorTape};

#[derive(Debug, Clone)]
pub struct Backend {
    /// Operators that the device is able to execute.
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

async fn run(backend: Backend, receiver: flume::Receiver<DeviceEvent>) {
    while let Ok(event) = receiver.recv_async().await {
        match event {
            DeviceEvent::Execute(tape) | DeviceEvent::ExecuteRead(tape, _) => {
                let this = tape.this;
                match execute_tape(&backend, tape) {
                    Ok(_) => log::info!("[run] {this}"),
                    Err(err) => log::error!("[run] {err}"),
                }
            }
        }
    }
}

fn execute_tape(backend: &Backend, tape: TensorTape) -> Result<(), Box<dyn std::error::Error>> {
    let mut allocator = Allocator::default();
    let ops = tape
        .ops
        .into_iter()
        .map(|op| allocator.alloc(op))
        .collect::<Result<Vec<_>, _>>()?;
    for op in ops {
        let io = op.io();
        backend.execute(op, io);
    }
    Ok(())
}
