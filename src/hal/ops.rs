use std::marker::PhantomData;

use mia_derive::TensorOp;

use crate::loom::{
    num::Scalar,
    ops::{TensorIr, TensorOp, TensorOpId},
};

#[derive(Debug, Clone)]
pub struct BinaryOp {
    pub id: TensorOpId,
    pub inputs: [TensorIr; 2],
    pub output: TensorIr,
}

impl TensorOp for BinaryOp {
    fn id(&self) -> TensorOpId {
        self.id
    }

    fn io(&self) -> Vec<TensorIr> {
        [&self.inputs[0], &self.inputs[1], &self.output]
            .into_iter()
            .cloned()
            .collect()
    }
}

#[derive(Debug, Clone, TensorOp)]
#[tensor_op(crate = "crate", bound = "T: Scalar")]
pub struct AddOp<T> {
    #[tensor_op]
    pub op: BinaryOp,
    pub phantom: PhantomData<T>,
}

impl<T: Scalar> From<BinaryOp> for AddOp<T> {
    fn from(value: BinaryOp) -> Self {
        Self {
            op: value,
            phantom: PhantomData,
        }
    }
}
