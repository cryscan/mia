use std::{marker::PhantomData, sync::Arc};

use mia_derive::TensorOp;

use crate::loom::{
    device::Backend,
    num::Scalar,
    ops::{BackendOp, InnerOp, TensorIr},
};

#[derive(Debug, Clone, TensorOp)]
#[tensor_op(crate = "crate")]
pub struct CreateOp {
    #[tensor_op]
    pub op: InnerOp<0, 1>,
    pub contents: Arc<[u8]>,
}

impl<B: Backend> BackendOp<B> for CreateOp {
    async fn execute(&self, backend: &mut B, io: Vec<TensorIr>) {
        backend.create(io[0].id, &self.contents);
    }
}

#[derive(Debug, Clone, TensorOp)]
#[tensor_op(crate = "crate", bound = "T: Scalar")]
pub struct AddOp<T> {
    #[tensor_op]
    pub op: InnerOp<2, 1>,
    pub phantom: PhantomData<T>,
}

impl<T: Scalar> From<InnerOp<2, 1>> for AddOp<T> {
    fn from(value: InnerOp<2, 1>) -> Self {
        Self {
            op: value,
            phantom: PhantomData,
        }
    }
}
