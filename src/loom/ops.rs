use std::any::Any;

use derive_more::Display;
use serde::{Deserialize, Serialize};

use super::{
    device::Device,
    layout::Layout,
    num::{DataType, Scalar},
    tensor::Tensor,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Display, Serialize, Deserialize)]
pub enum Access {
    ReadOnly,
    ReadWrite,
    WriteOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorIr {
    pub layout: Layout,
    pub r#type: DataType,
    pub id: usize,
    pub count: usize,
    pub access: Access,
}

impl<D: Device, T: Scalar> Tensor<D, T> {
    #[inline]
    pub fn ir(&self, access: Access) -> TensorIr {
        let layout = self.layout();
        let r#type = T::DATA_TYPE;
        let count = self.ref_count();
        let id = self.id().get();
        TensorIr {
            layout,
            r#type,
            id,
            count,
            access,
        }
    }
}

pub trait TensorOp: Send + Sync + Any {
    /// Input and output tensors of the op.
    fn io(&self) -> Vec<TensorIr>;
}

impl std::ops::Deref for dyn TensorOp {
    type Target = dyn Any;

    fn deref(&self) -> &Self::Target {
        self as &dyn Any
    }
}

impl std::ops::DerefMut for dyn TensorOp {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self as &mut dyn Any
    }
}

impl From<Box<dyn TensorOp>> for Box<dyn Any> {
    fn from(value: Box<dyn TensorOp>) -> Self {
        let value = Box::leak(value);
        unsafe { Box::from_raw(value as *mut dyn Any) }
    }
}
