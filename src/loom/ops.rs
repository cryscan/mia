use std::{any::Any, borrow::Cow};

use derive_more::{Deref, DerefMut, Display};
use dyn_clone::DynClone;
use serde::{Deserialize, Serialize};

use super::{
    device::Device,
    layout::Layout,
    num::{DataType, Scalar},
    tensor::{Tensor, TensorId},
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
    pub id: TensorId,
    pub count: usize,
    pub access: Access,
}

impl TensorIr {
    #[inline]
    pub fn data_count(&self) -> usize {
        self.layout.size() * self.r#type.count()
    }

    #[inline]
    pub fn data_size(&self) -> usize {
        self.layout.size() * self.r#type.size()
    }

    #[inline]
    pub fn is_compatible(&self, other: &TensorIr) -> bool {
        self.r#type == other.r#type && self.data_count() == other.data_count()
    }
}

impl<D: Device, T: Scalar> Tensor<D, T> {
    #[inline]
    pub fn ir(&self, access: Access) -> TensorIr {
        let layout = self.layout();
        let r#type = T::DATA_TYPE;
        let count = self.ref_count();
        let id = self.id();
        TensorIr {
            layout,
            r#type,
            id,
            count,
            access,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Deref, DerefMut)]
pub struct TensorOpId(uid::Id<TensorOpId>);

pub trait TensorOp: DynClone + Send + Sync + Any {
    /// Name of the op, by default it's the type name.
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed(std::any::type_name::<Self>())
    }

    /// Id of the op.
    fn id(&self) -> TensorOpId;
    /// Input and output tensors of the op.
    fn io(&self) -> Vec<TensorIr>;
}

dyn_clone::clone_trait_object!(TensorOp);

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

impl std::fmt::Debug for dyn TensorOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorOp")
            .field("name", &self.name())
            .field("id", &self.id())
            .field("io", &self.io())
            .finish()
    }
}

/// Records operators a tensor has experienced.
#[derive(Debug, Default, Clone)]
pub struct TensorTape {
    /// The ID of the tensor itself.
    pub id: TensorId,
    /// Operators the tensor has experienced.
    pub ops: Vec<Box<dyn TensorOp>>,
}

impl PartialEq for TensorTape {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for TensorTape {}
