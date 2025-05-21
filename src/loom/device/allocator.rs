use std::borrow::Cow;

use itertools::Itertools;
use rustc_hash::FxHashMap as HashMap;

use super::{Device, DeviceOp};
use crate::loom::{
    ops::{Access, TensorIr, TensorOp},
    tensor::TensorId,
};

#[allow(unused)]
#[derive(Debug, Default, Clone)]
pub struct Allocator {
    map: HashMap<TensorId, TensorId>,
    free: HashMap<usize, Vec<TensorId>>,
}

impl Allocator {
    pub fn alloc(&mut self, op: Box<dyn TensorOp>) -> AllocatedOp {
        let io = op.io();
        let mut io_ref = io.iter().collect_vec();

        // collect input tensors that can be immediately reused
        let mut short_free = io
            .iter()
            .filter(|ir| matches!(ir.access, Access::ReadOnly))
            .filter(|ir| ir.count <= 1)
            .collect_vec();

        for ir in io_ref
            .iter_mut()
            .filter(|ir| matches!(ir.access, Access::WriteOnly))
        {
            if let Some((i, _)) = short_free
                .iter()
                .find_position(|x| x.data_size() == ir.data_size())
            {
                *ir = short_free.remove(i);
            }
        }

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
    fn name(&self) -> Cow<'static, str> {
        let type_name = std::any::type_name::<Self>();
        let op_name = self.op.name();
        Cow::from(format!("{type_name}<{op_name}>"))
    }

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
