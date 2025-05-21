use derive_more::{Deref, DerefMut};
use rustc_hash::FxHashMap as HashMap;

use super::{Device, DeviceOp};
use crate::loom::ops::{Access, TensorIr, TensorOp};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Deref, DerefMut)]
struct TensorId(usize);

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Data {
    id: usize,
    size: usize,
}

#[allow(unused)]
#[derive(Debug, Default, Clone)]
pub struct Allocator {
    map: HashMap<TensorId, Data>,
    free: Vec<Data>,
}

impl Allocator {
    pub fn alloc(&mut self, op: Box<dyn TensorOp>) -> AllocatedOp {
        let io = op.io();
        let (_input, _output): (Vec<_>, Vec<_>) = io
            .into_iter()
            .partition(|ir| matches!(ir.access, Access::ReadOnly));

        todo!()
    }
}

/// A wrapper around another [`TensorOp`], of which storages are optimized by the allocator.
pub struct AllocatedOp {
    op: Box<dyn TensorOp>,
    io: Vec<TensorIr>,
}

impl TensorOp for AllocatedOp {
    #[inline]
    fn io(&self) -> Vec<TensorIr> {
        self.io.clone()
    }
}

impl<D: Device> DeviceOp<AllocatedOp> for D {
    #[inline]
    fn execute(&self, op: AllocatedOp, io: Vec<TensorIr>) {
        self.execute_dyn(op.op, io);
    }
}
