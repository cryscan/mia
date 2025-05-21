use std::any::TypeId;

use rustc_hash::FxHashMap as HashMap;
use serde::{Deserialize, Serialize};

use super::ops::{TensorIr, TensorOp};

pub use cpu::{Cpu, CpuBuilder};
pub use gpu::{Gpu, GpuBuilder};

pub mod allocator;
pub mod cpu;
pub mod gpu;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId;

/// Implemented for each [`Device`] for each [`TensorOp`].
pub trait DeviceOp<Op: TensorOp> {
    /// Statically dispatch to actual `op`'s execution, given arguments `io`.
    fn execute(&self, op: Op, io: Vec<TensorIr>);
}

pub trait Device {
    /// Dynamically dispatch to actual `op`'s execution, given arguments `io`.
    fn execute_dyn(&self, op: Box<dyn TensorOp>, io: Vec<TensorIr>);

    /// Dynamically dispatch to actual `op`'s execution, using `op`'s own `io`.
    #[inline]
    fn execute_op_dyn(&self, op: Box<dyn TensorOp>) {
        let io = op.io();
        self.execute_dyn(op, io);
    }
}

type OpVTable<D> = HashMap<TypeId, fn(&D, Box<dyn TensorOp>, Vec<TensorIr>)>;

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::{Cpu, CpuBuilder, Device, DeviceOp};
    use crate::loom::ops::{TensorIr, TensorOp};

    #[test]
    fn test_add_op() {
        struct PhonyOp<const N: usize>;

        impl<const N: usize> TensorOp for PhonyOp<N> {
            fn name(&self) -> Cow<'static, str> {
                Cow::from(std::any::type_name::<Self>())
            }

            fn io(&self) -> Vec<TensorIr> {
                vec![]
            }
        }

        impl<const N: usize> DeviceOp<PhonyOp<N>> for Cpu {
            fn execute(&self, op: PhonyOp<N>, _io: Vec<TensorIr>) {
                println!("execute op: {}", op.name());
            }
        }

        let cpu = CpuBuilder::new()
            .add_op::<PhonyOp<0>>()
            .add_op::<PhonyOp<1>>()
            .add_op::<PhonyOp<2>>()
            .add_op::<PhonyOp<3>>()
            .build();
        let ops: Vec<Box<dyn TensorOp>> = vec![
            Box::new(PhonyOp::<3>),
            Box::new(PhonyOp::<2>),
            Box::new(PhonyOp::<1>),
            Box::new(PhonyOp::<0>),
        ];
        ops.into_iter().for_each(|op| cpu.execute_op_dyn(op));
    }
}
