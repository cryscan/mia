use std::any::TypeId;

use rustc_hash::FxHashMap as HashMap;
use serde::{Deserialize, Serialize};

use super::ops::TensorOp;

pub use cpu::{Cpu, CpuBuilder};
pub use gpu::{Gpu, GpuBuilder};

pub mod allocator;
pub mod cpu;
pub mod gpu;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId;

/// Implemented for each [`Device`] for each [`TensorOp`].
/// Defines an `op`'s actual execution on the device.
pub trait DeviceOp<Op: TensorOp> {
    fn execute(&self, op: Op);
}

pub trait Device {
    fn execute_dyn(&self, op: Box<dyn TensorOp>);
}

type OpVTable<D> = HashMap<TypeId, fn(&D, Box<dyn TensorOp>)>;

#[cfg(test)]
mod tests {
    use super::{Cpu, CpuBuilder, Device, DeviceOp};
    use crate::loom::ops::{TensorIr, TensorOp};

    #[test]
    fn test_add_op() {
        struct PhonyOp<const N: usize>;

        impl<const N: usize> TensorOp for PhonyOp<N> {
            fn io(&self) -> Vec<TensorIr> {
                vec![]
            }
        }

        impl<const N: usize> DeviceOp<PhonyOp<N>> for Cpu {
            fn execute(&self, _op: PhonyOp<N>) {
                println!("execute phony op: {N}");
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
        ops.into_iter().for_each(|op| cpu.execute_dyn(op));
    }
}
