use std::{borrow::Cow, sync::Arc};

use itertools::Itertools;

use super::ops::{AddOp, CreateOp};
use crate::loom::{
    device::{Device, DeviceError, DeviceEvent},
    layout::IntoLayout,
    num::Scalar,
    ops::{Access, InnerOp, TensorOp},
    tensor::{Tensor, TensorError},
};

fn binary_op_unchecked<D, T, Op>(lhs: Tensor<D, T>, rhs: Tensor<D, T>) -> Tensor<D, T>
where
    D: Device,
    T: Scalar,
    Op: From<InnerOp<2, 1>> + TensorOp,
{
    let mut output = Tensor::zeros_like(&lhs);
    let mut ops = [lhs.as_ref().ops.clone(), rhs.as_ref().ops.clone()]
        .concat()
        .into_iter()
        .unique()
        .collect_vec();

    let inputs = [lhs.ir(Access::ReadOnly), rhs.ir(Access::ReadOnly)];
    let outputs = [output.ir(Access::WriteOnly)];
    let op = Op::from(InnerOp::new(inputs, outputs));
    ops.push(Box::new(op));

    _ = output.replace_ops(ops);

    output
}

impl<D: Device, T: Scalar> Tensor<D, T> {
    /// Replace the inner ops recorded on the tape of the tensor. Panics if the tensor isn't unique.
    fn replace_ops(&mut self, ops: Vec<Box<dyn TensorOp>>) -> Vec<Box<dyn TensorOp>> {
        let tape = self.as_mut();
        let tape = Arc::get_mut(tape).expect("must be unique");
        std::mem::replace(&mut tape.ops, ops)
    }

    pub fn create(
        device: Arc<D>,
        layout: impl IntoLayout,
        contents: impl Into<Cow<'static, [T]>>,
    ) -> Result<Self, TensorError> {
        let layout = layout.into_layout();
        let contents: Cow<'static, [T]> = contents.into();
        if layout.co_size() > contents.len() {
            return Err(TensorError::Create(layout, contents.len()));
        }

        let mut output = Tensor::<D, T>::zeros(device, layout);
        let contents = match contents {
            Cow::Borrowed(contents) => bytemuck::cast_slice(contents).into(),
            Cow::Owned(contents) => bytemuck::cast_slice(&contents).to_vec().into(),
        };
        let op = InnerOp::new([], [output.ir(Access::WriteOnly)]);
        let op = CreateOp { op, contents };
        let ops: Vec<Box<dyn TensorOp>> = vec![Box::new(op)];

        _ = output.replace_ops(ops);

        Ok(output)
    }

    #[inline]
    pub fn try_add(self, rhs: Tensor<D, T>) -> Result<Self, TensorError> {
        if self.layout() != rhs.layout() {
            return Err(TensorError::Layout(self.layout(), rhs.layout()));
        }
        Ok(binary_op_unchecked::<_, _, AddOp<T>>(self, rhs))
    }

    #[inline]
    pub async fn back(self) -> Result<Box<[T]>, DeviceError> {
        let (sender, receiver) = flume::bounded(0);
        let tape = self.tape().clone();
        let event = DeviceEvent::Back { tape, sender };
        self.device().execute(event);

        let data = receiver.recv_async().await??;
        Ok(bytemuck::cast_slice(&data.0).to_vec().into_boxed_slice())
    }
}

impl<D: Device, T: Scalar> std::ops::Add<Tensor<D, T>> for Tensor<D, T> {
    type Output = Tensor<D, T>;

    fn add(self, rhs: Tensor<D, T>) -> Self::Output {
        self.try_add(rhs).unwrap()
    }
}
