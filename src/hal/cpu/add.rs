use half::f16;

use crate::{
    hal::ops::AddOp,
    loom::{
        device::cpu::Backend,
        ops::{BackendOp, TensorIr},
    },
};

impl BackendOp<Backend> for AddOp<f32> {
    async fn execute(&self, _backend: &mut Backend, _io: Vec<TensorIr>) {
        todo!()
    }
}

impl BackendOp<Backend> for AddOp<f16> {
    async fn execute(&self, _backend: &mut Backend, _io: Vec<TensorIr>) {
        todo!()
    }
}
