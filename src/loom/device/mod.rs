use std::any::TypeId;

use derive_more::{Deref, DerefMut};
use rustc_hash::FxHashMap as HashMap;

use super::ops::{TensorIr, TensorOp, TensorTape};

pub use cpu::{Cpu, CpuBuilder};
pub use gpu::{Gpu, GpuBuilder};

pub mod allocator;
pub mod cpu;
pub mod gpu;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Deref, DerefMut)]
pub struct DeviceId(uid::Id<DeviceId>);

pub trait Device {
    /// Dynamically dispatch to actual `op`'s execution, using `op`'s own `io`.
    fn execute(&self, event: DeviceEvent);
}

#[derive(Debug, Clone)]
pub enum DeviceEvent {
    Execute(TensorTape),
    ExecuteRead(TensorTape, flume::Sender<Box<[u8]>>),
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
    use std::time::Duration;

    use super::{BackendOp, CpuBuilder, Device, DeviceEvent, cpu};
    use crate::loom::ops::{TensorIr, TensorOp, TensorOpId, TensorTape};

    #[tokio::test]
    async fn test_add_op() {
        #[derive(Debug, Default, Clone)]
        struct PhonyOp<const N: usize>(TensorOpId);

        impl<const N: usize> TensorOp for PhonyOp<N> {
            fn id(&self) -> TensorOpId {
                self.0
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
            Box::new(PhonyOp::<3>::default()),
            Box::new(PhonyOp::<2>::default()),
            Box::new(PhonyOp::<1>::default()),
            Box::new(PhonyOp::<0>::default()),
        ];
        let this = Default::default();
        let tape = TensorTape { this, ops };
        cpu.execute(DeviceEvent::Execute(tape));

        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}
