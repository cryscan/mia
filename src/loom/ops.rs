use derive_more::Display;
use serde::{Deserialize, Serialize};

use super::{device::Device, layout::Layout, num::DataType, tensor::TensorUntyped};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Display, Serialize, Deserialize)]
pub enum Access {
    ReadOnly,
    ReadWrite,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorIr {
    pub layout: Layout,
    pub r#type: DataType,
    pub id: usize,
    pub access: Access,
}

impl TensorIr {
    /// Override the access of the IR.
    #[inline]
    pub fn update(&mut self, access: Access) {
        self.access = self.access.max(access);
    }
}

impl<D: Device> TensorUntyped<D> {
    #[inline]
    pub fn ir(&self, access: Access) -> TensorIr {
        let layout = self.layout();
        let r#type = self.data_type();
        let id = self.id().get();
        TensorIr {
            layout,
            r#type,
            id,
            access,
        }
    }
}

pub trait TensorOp: Send + Sync {
    /// Iterates through input and output tensors.
    fn io(&self) -> (Vec<&TensorIr>, Vec<&TensorIr>);
    /// Iterates through input and output tensors (mutable).
    fn io_mut(&mut self) -> (Vec<&mut TensorIr>, Vec<&mut TensorIr>);

    /// Set the tensors that would be written to as [`Access::ReadWrite`].
    fn update_io_access(&mut self) {
        let (mut inputs, outputs) = self.io_mut();
        for output in outputs {
            output.update(Access::ReadWrite);
            if let Some(input) = inputs.iter_mut().find(|input| input.id == output.id) {
                input.update(Access::ReadWrite);
            }
        }
    }
}
