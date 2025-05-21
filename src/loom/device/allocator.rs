use derive_more::{Deref, DerefMut};
use rustc_hash::FxHashMap as HashMap;

use super::{Device, DeviceOp};
use crate::loom::ops::{TensorIr, TensorOp};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Deref, DerefMut)]
struct TensorId(usize);

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Deref, DerefMut)]
struct DataId(usize);

#[allow(unused)]
#[derive(Debug, Default, Clone)]
pub struct Allocator {
    map: HashMap<TensorId, DataId>,
    free: Vec<DataId>,
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
    fn execute(&self, op: AllocatedOp) {
        self.execute_dyn(op.op);
    }
}
