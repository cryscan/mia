use std::{
    any::{Any, TypeId},
    sync::Arc,
};

use super::{Device, DeviceId, DeviceOp, OpVTable, allocator::AllocatedOp};
use crate::loom::ops::{TensorIr, TensorOp};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cpu {
    /// The unique identifier of the device.
    pub id: uid::Id<DeviceId>,
    /// Operators that the device is able to execute.
    pub ops: Arc<OpVTable<Self>>,
}

impl Device for Cpu {
    #[inline]
    fn execute_dyn(&self, op: Box<dyn TensorOp>, io: Vec<TensorIr>) {
        assert_eq!(op.io().len(), io.len());
        let id = &op.as_ref().type_id();
        let name = std::any::type_name_of_val(op.as_ref());
        match self.ops.get(id) {
            Some(f) => f(self, op, io),
            None => log::error!("unable to execute op of type {}", name),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct CpuBuilder {
    pub ops: OpVTable<Cpu>,
}

impl CpuBuilder {
    pub fn new() -> Self {
        Self::default().add_op::<AllocatedOp>()
    }

    pub fn build(self) -> Cpu {
        let id = uid::Id::new();
        let ops = self.ops.into();
        Cpu { id, ops }
    }

    pub fn add_op<Op>(mut self) -> Self
    where
        Op: TensorOp,
        Cpu: DeviceOp<Op>,
    {
        let id = TypeId::of::<Op>();
        let f = |cpu: &Cpu, op: Box<dyn TensorOp>, io| match Box::<dyn Any>::from(op).downcast() {
            Ok(op) => cpu.execute(*op, io),
            Err(_) => unreachable!(),
        };
        self.ops.insert(id, f);
        self
    }
}
