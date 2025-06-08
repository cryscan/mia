use std::any::TypeId;

use derive_more::{Deref, DerefMut};
use rustc_hash::FxHashMap as HashMap;
use thiserror::Error;

use super::{
    ops::{TensorIr, TensorOp, TensorTape},
    platform::BoxFuture,
    tensor::TensorId,
};

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

#[derive(Debug, Error)]
pub enum DeviceError {
    #[error("failed to map buffer")]
    Buffer(#[from] wgpu::BufferAsyncError),
    #[error("failed to receive")]
    Recv(#[from] flume::RecvError),
    #[error("failed to allocate tensor")]
    Alloc(#[from] allocator::AllocError),
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct BackData(pub Box<[u8]>);

#[derive(Debug, Clone)]
pub enum DeviceEvent {
    Execute {
        tape: TensorTape,
        sender: flume::Sender<Result<TensorId, DeviceError>>,
    },
    Back {
        tape: TensorTape,
        sender: flume::Sender<Result<BackData, DeviceError>>,
    },
    Cleanup {
        retain: Vec<TensorTape>,
    },
}

#[cfg_attr(not(target_arch = "wasm32"), trait_variant::make(Send))]
pub trait Backend {
    type Data;

    /// Dynamically dispatch to actual `op`'s execution, using given `io`.
    async fn execute(&mut self, op: &dyn TensorOp, io: Vec<TensorIr>);
    /// Create a buffer for tensor of `id`.
    fn create(&mut self, id: TensorId, contents: &[u8]) -> Self::Data;
    /// Allocate a buffer for tensor of `id`.
    fn alloc(&mut self, id: TensorId, size: usize) -> Self::Data;
    /// Get the buffer of tensor of `id`.
    fn fetch(&self, id: TensorId) -> Self::Data;
}

type OpFn<B> = for<'a> fn(&'a mut B, &'a dyn TensorOp, Vec<TensorIr>) -> BoxFuture<'a, ()>;
type OpVTable<B> = HashMap<TypeId, OpFn<B>>;

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::{CpuBuilder, Device, DeviceEvent, cpu};
    use crate::loom::ops::{BackendOp, TensorIr, TensorOp, TensorOpId, TensorTape};

    #[tokio::test]
    async fn test_add_op() -> Result<(), Box<dyn Error>> {
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

        impl<const N: usize> BackendOp<cpu::Backend> for PhonyOp<N> {
            async fn execute(&self, _backend: &mut cpu::Backend, _io: Vec<TensorIr>) {
                println!("{}", self.name())
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
        let id = Default::default();
        let tape = TensorTape { id, ops };
        let (sender, receiver) = flume::bounded(0);
        cpu.execute(DeviceEvent::Execute { tape, sender });

        let _ = receiver.recv_async().await??;
        Ok(())
    }
}
