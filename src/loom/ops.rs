use std::{any::Any, borrow::Cow};

use derive_more::{Deref, DerefMut, Display};
use dyn_clone::DynClone;

use mia_derive::TensorOp;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{
    device::{Backend, Device},
    layout::{IntoLayout, Layout},
    num::{DataType, Scalar},
    slice::Slice,
    tensor::{Tensor, TensorId},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Display)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Access {
    ReadOnly,
    ReadWrite,
    WriteOnly,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TensorIr {
    pub layout: Layout,
    pub slice: Slice,
    pub r#type: DataType,
    pub id: TensorId,
    pub count: usize,
    pub access: Access,
}

impl TensorIr {
    /// Creates a unique IR of the tensor.
    ///
    /// ## Safety
    /// This function is safe to call iff:
    /// 1. The given `id` and `layout` originate from the same tensor;
    /// 2. The tensor has exactly one active reference.
    pub unsafe fn unique<T: Scalar>(id: TensorId, layout: impl IntoLayout, access: Access) -> Self {
        let layout = layout.into_layout();
        let slice = Slice::from_layout(layout.clone());
        let r#type = T::DATA_TYPE;
        let count = 1;
        Self {
            layout,
            slice,
            r#type,
            id,
            count,
            access,
        }
    }

    #[inline]
    pub fn data_len(&self) -> usize {
        self.layout.size() * self.r#type.count()
    }

    #[inline]
    pub fn data_size(&self) -> usize {
        self.layout.size() * self.r#type.size()
    }

    #[inline]
    pub fn is_compatible(&self, other: &TensorIr) -> bool {
        self.r#type == other.r#type && self.data_len() == other.data_len()
    }
}

impl<D: Device, T: Scalar> Tensor<D, T> {
    #[inline]
    pub fn ir(&self, access: Access) -> TensorIr {
        let layout = self.layout();
        let slice = Slice::from_layout(layout.clone());
        let r#type = T::DATA_TYPE;
        let count = self.ref_count();
        let id = self.id();
        TensorIr {
            layout,
            slice,
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

impl PartialEq for dyn TensorOp {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for dyn TensorOp {}

impl std::hash::Hash for dyn TensorOp {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id().hash(state);
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

/// Implemented for each [`Device`] for each [`TensorOp`].
#[cfg_attr(not(target_arch = "wasm32"), trait_variant::make(Send))]
pub trait BackendOp<B: Backend> {
    async fn execute(&self, backend: &mut B, io: Vec<TensorIr>);
}

/// Records operators a tensor has experienced.
#[derive(Debug, Clone)]
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

#[derive(Debug, Display, Clone, Deref, DerefMut)]
pub struct Mermaid(pub String);

impl TensorTape {
    /// Prints the tape in Mermaid format.
    pub fn print_mermaid(&self) -> Mermaid {
        let mut s = "graph TD\n".to_string();

        for (index, op) in self.ops.iter().enumerate() {
            let op_node = format!("op_{}", index);
            let op_label = op.name();
            s.push_str(&format!("    {}[\"{}\"]\n", op_node, op_label));

            for ir in op.io() {
                let tensor_node = format!("tensor_{}", ir.id);
                let tensor_label = format!("{}", ir.id);

                match ir.access {
                    Access::ReadOnly => {
                        s.push_str(&format!("    {}((\"{}\"))\n", tensor_node, tensor_label));
                        s.push_str(&format!("    {} --> |Read| {}\n", tensor_node, op_node));
                    }
                    Access::WriteOnly => {
                        s.push_str(&format!("    {}((\"{}\"))\n", tensor_node, tensor_label));
                        s.push_str(&format!("    {} --> |Write| {}\n", op_node, tensor_node));
                    }
                    _ => {}
                }
            }
        }
        Mermaid(s)
    }
}

#[derive(Debug, Clone)]
pub struct InnerOp<const I: usize, const O: usize> {
    pub id: TensorOpId,
    pub inputs: [TensorIr; I],
    pub outputs: [TensorIr; O],
}

impl<const I: usize, const O: usize> InnerOp<I, O> {
    #[inline]
    pub fn new(inputs: [TensorIr; I], outputs: [TensorIr; O]) -> Self {
        inputs
            .iter()
            .for_each(|ir| assert_eq!(ir.access, Access::ReadOnly));
        outputs
            .iter()
            .for_each(|ir| assert_eq!(ir.access, Access::WriteOnly));
        let id = Default::default();
        Self {
            id,
            inputs,
            outputs,
        }
    }
}

impl<const I: usize, const O: usize> TensorOp for InnerOp<I, O> {
    fn id(&self) -> TensorOpId {
        self.id
    }

    fn io(&self) -> Vec<TensorIr> {
        self.inputs
            .iter()
            .chain(self.outputs.iter())
            .cloned()
            .collect()
    }
}

/// The initial op, which is inserted as the first op in a tensor's tape.
/// the allocator can then ensure that the tensor's buffer gets allocated on initialization.
#[derive(Debug, Clone, TensorOp)]
#[tensor_op(crate = "crate")]
pub struct ZeroOp(pub InnerOp<0, 1>);

impl ZeroOp {
    #[inline]
    pub fn new(ir: TensorIr) -> Self {
        Self(InnerOp::new([], [ir]))
    }
}

impl<B: Backend> BackendOp<B> for ZeroOp {
    async fn execute(&self, _: &mut B, _: Vec<TensorIr>) {}
}

/// The terminal op, which is inserted after the tensor is consumed.
/// The allocator can then have the chance to reclaim it.
#[derive(Debug, Clone, TensorOp)]
#[tensor_op(crate = "crate")]
pub struct OneOp(pub InnerOp<1, 0>);

impl OneOp {
    #[inline]
    pub fn new(ir: TensorIr) -> Self {
        Self(InnerOp::new([ir], []))
    }
}

impl<B: Backend> BackendOp<B> for OneOp {
    async fn execute(&self, _: &mut B, _: Vec<TensorIr>) {}
}
