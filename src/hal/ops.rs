use std::{marker::PhantomData, sync::Arc};

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

macro_rules! impl_op_from_inner {
    ($op:ty, $f:ty) => {
        impl From<$f> for $op {
            fn from(op: $f) -> Self {
                let phantom = PhantomData;
                Self { op, phantom }
            }
        }
    };
    ($op:ty, $f:ty, $($t:ident: $b:ident),+) => {
        impl<$($t: $b),+> From<$f> for $op {
            fn from(op: $f) -> Self {
                let phantom = PhantomData;
                Self { op, phantom }
            }
        }
    };
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
        let contents = bytemuck::cast_slice(&self.contents);
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

impl_op_from_inner!(AddOp<T>, InnerOp<2, 1>, T: Scalar);

#[derive(Debug, Clone, TensorOp)]
#[tensor_op(crate = "crate", bound = "T: Scalar")]
pub struct LayerNormOp<T> {
    #[tensor_op]
    pub op: InnerOp<1, 1>,
    pub phantom: PhantomData<T>,
}

impl_op_from_inner!(LayerNormOp<T>, InnerOp<1, 1>, T: Scalar);
