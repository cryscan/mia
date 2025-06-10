use derive_more::{Deref, DerefMut};

use crate::loom::{
    device,
    ops::{BackendOp, TensorIr, TensorOp, TensorOpId},
};

#[derive(Debug, Clone)]
pub struct BinaryOp {
    pub id: TensorOpId,
    pub inputs: [TensorIr; 2],
    pub output: TensorIr,
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct AddOp(pub BinaryOp);

impl TensorOp for AddOp {
    fn id(&self) -> TensorOpId {
        self.id
    }

    fn io(&self) -> Vec<TensorIr> {
        vec![
            self.inputs[0].clone(),
            self.inputs[1].clone(),
            self.output.clone(),
        ]
    }
}

impl BackendOp<device::cpu::Backend> for AddOp {
    async fn execute(&self, _backend: &mut device::cpu::Backend, _io: Vec<TensorIr>) {
        todo!()
    }
}
