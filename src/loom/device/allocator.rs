use std::{borrow::Cow, cell::RefCell, rc::Rc};

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
        let io = op
            .io()
            .into_iter()
            .map(RefCell::new)
            .map(Rc::new)
            .collect_vec();

        // tensors that can be immediately reused
        let mut free = io
            .iter()
            .filter(|ir| matches!(ir.borrow().access, Access::ReadOnly))
            .filter(|ir| ir.borrow().count <= 1)
            .cloned()
            .collect_vec();
        for x in io
            .iter()
            .filter(|ir| matches!(ir.borrow().access, Access::WriteOnly))
        {
            if let Some((index, _)) = free
                .iter()
                .find_position(|y| y.borrow().is_compatible(&x.borrow()))
            {
                let y = free.remove(index);
                x.borrow_mut().id = y.borrow().id;
                x.borrow_mut().access = Access::ReadWrite;
                y.borrow_mut().access = Access::ReadWrite;
            }
        }

        // reuse tensors from the global free list by building a redirection map
        for x in io
            .iter()
            .filter(|ir| matches!(ir.borrow().access, Access::WriteOnly))
        {
            if let Some(free) = self
                .free
                .get_mut(&x.borrow().data_size())
                .and_then(|free| free.pop())
            {
                self.map.insert(x.borrow().id, free);
            }
        }

        // substitute tensor ids following the redirection map
        for ir in io.iter() {
            let id = ir.borrow().id;
            let id = match self.map.get(&id) {
                Some(&id) => id,
                None => id,
            };
            ir.borrow_mut().id = id;
        }

        // expand the free list
        for ir in free {
            let size = ir.borrow().data_size();
            let id = ir.borrow().id;
            if let Some(ids) = self.free.get_mut(&size) {
                ids.push(id);
            } else {
                self.free.insert(size, vec![id]);
            }
        }

        let io = io.into_iter().map(|ir| ir.borrow().clone()).collect();
        AllocatedOp { op, io }
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
