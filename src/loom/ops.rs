use derive_more::Display;
use serde::{Deserialize, Serialize};

use super::{device::Device, layout::Layout, num::DataType, tensor::TensorUntyped};

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
    pub access: Access,
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
    /// Input and output tensors.
    fn io(&self) -> (Vec<&TensorIr>, Vec<&TensorIr>);
    /// Input and output tensors (mutable).
    fn io_mut(&mut self) -> (Vec<&mut TensorIr>, Vec<&mut TensorIr>);
}

/// Set the tensors that would be written to as [`Access::ReadWrite`].
#[allow(unused)]
fn update_io_access(op: &mut impl TensorOp) {
    let (mut inputs, outputs) = op.io_mut();
    for output in outputs {
        match inputs.iter_mut().find(|input| input.id == output.id) {
            Some(input) => {
                input.access = Access::ReadWrite;
                output.access = Access::ReadWrite;
            }
            None => output.access = Access::WriteOnly,
        }
    }
}
