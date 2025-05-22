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

pub trait Device {
    /// Dynamically dispatch to actual `op`'s execution, using `op`'s own `io`.
    fn execute(&self, op: Box<dyn TensorOp>);
}

/// Implemented for each [`Device`] for each [`TensorOp`].
pub trait BackendOp<Op: TensorOp> {
    /// Statically dispatch to actual `op`'s execution, using given `io`.
    fn execute(&self, op: Op, io: Vec<TensorIr>);
}

pub trait Backend: Send + Sync {
    /// Dynamically dispatch to actual `op`'s execution, using given `io`.
    fn execute(&self, op: Box<dyn TensorOp>, io: Vec<TensorIr>);
}

type OpVTable<B> = HashMap<TypeId, fn(&B, Box<dyn TensorOp>, Vec<TensorIr>)>;

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod tests {
    use std::{borrow::Cow, time::Duration};

    use super::{BackendOp, CpuBuilder, Device, cpu};
    use crate::loom::ops::{TensorIr, TensorOp};

    #[tokio::test]
    async fn test_add_op() {
        struct PhonyOp<const N: usize>;

        impl<const N: usize> TensorOp for PhonyOp<N> {
            fn name(&self) -> Cow<'static, str> {
                Cow::from(std::any::type_name::<Self>())
            }

            fn io(&self) -> Vec<TensorIr> {
                vec![]
            }
        }

        impl<const N: usize> BackendOp<PhonyOp<N>> for cpu::Backend {
            fn execute(&self, op: PhonyOp<N>, _io: Vec<TensorIr>) {
                println!("{}", op.name());
            }
        }

        let cpu = CpuBuilder::new()
            .add_op::<PhonyOp<0>>()
            .add_op::<PhonyOp<1>>()
            .add_op::<PhonyOp<2>>()
            .add_op::<PhonyOp<3>>()
            .build()
            .await;
        let ops: Vec<Box<dyn TensorOp>> = vec![
            Box::new(PhonyOp::<3>),
            Box::new(PhonyOp::<2>),
            Box::new(PhonyOp::<1>),
            Box::new(PhonyOp::<0>),
        ];
        ops.into_iter().for_each(|op| cpu.execute(op));

        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}
