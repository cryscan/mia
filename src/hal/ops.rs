use std::{borrow::Cow, marker::PhantomData, sync::Arc};

use half::f16;
use mia_derive::TensorOp;

use crate::loom::{
    device::{Backend, CpuBuilder, GpuBuilder},
    num::{F16x4, F32x4, Scalar},
    ops::{BackendOp, InnerOp, OneOp, TensorIr, ZeroOp},
};

impl CpuBuilder {
    pub fn add_default_ops(self) -> Self {
        self.add_op::<ZeroOp>()
            .add_op::<OneOp>()
            .add_op::<CreateOp<f32>>()
            .add_op::<CreateOp<f16>>()
            .add_op::<AddOp<f32>>()
            .add_op::<AddOp<f16>>()
            .add_op::<AddOp<F32x4>>()
            .add_op::<AddOp<F16x4>>()
            .add_op::<SoftmaxOp<f32>>()
            .add_op::<SoftmaxOp<f16>>()
            .add_op::<LayerNormOp<f32>>()
            .add_op::<LayerNormOp<f16>>()
    }
}

impl GpuBuilder {
    pub fn add_default_ops(self) -> Self {
        self.add_op::<ZeroOp>()
            .add_op::<OneOp>()
            .add_op::<CreateOp<f32>>()
            .add_op::<CreateOp<f16>>()
    }
}

#[derive(Debug, Clone, TensorOp)]
#[tensor_op(crate = "crate")]
pub struct CreateOp<T: Scalar> {
    #[tensor_op]
    pub op: InnerOp<0, 1>,
    pub contents: Arc<[T]>,
}

impl<B: Backend, T: Scalar> BackendOp<B> for CreateOp<T> {
    async fn execute(&self, backend: &mut B, io: Vec<TensorIr>) {
        let contents = Cow::Borrowed(self.contents.as_ref());
        backend.create(io[0].id, contents);
    }
}

#[derive(Debug, Clone, TensorOp)]
#[tensor_op(crate = "crate", bound = "T: Scalar")]
pub struct AddOp<T> {
    #[tensor_op]
    pub op: InnerOp<2, 1>,
    pub phantom: PhantomData<T>,
}

#[derive(Debug, Clone, TensorOp)]
#[tensor_op(crate = "crate", bound = "T: Scalar")]
pub struct SoftmaxOp<T> {
    #[tensor_op]
    pub op: InnerOp<1, 1>,
    pub phantom: PhantomData<T>,
}

#[derive(Debug, Clone, TensorOp)]
#[tensor_op(crate = "crate", bound = "T: Scalar")]
pub struct LayerNormOp<T> {
    #[tensor_op]
    pub op: InnerOp<3, 1>,
    pub eps: f32,
    pub phantom: PhantomData<T>,
}

#[derive(Debug, Clone, TensorOp)]
#[tensor_op(crate = "crate", bound = "T: Scalar")]
pub struct MatMatFp16Op<T> {
    #[tensor_op]
    pub op: InnerOp<2, 1>,
    pub phantom: PhantomData<T>,
}

#[derive(Debug, Clone, TensorOp)]
#[tensor_op(crate = "crate", bound = "T: Scalar")]
pub struct MatVecFp16Op<T> {
    #[tensor_op]
    pub op: InnerOp<2, 1>,
    pub phantom: PhantomData<T>,
}
