use std::{marker::PhantomData, sync::Arc};

use half::f16;
use itertools::Itertools;

use super::ops::{AddOp, CreateOp, LayerNormOp, SoftmaxOp};
use crate::loom::{
    device::{Device, DeviceError, DeviceEvent},
    layout::IntoLayout,
    num::{Float, Scalar},
    ops::{Access, InnerOp, OneOp, TensorOp},
    tensor::Tensor,
};

fn binary_op_unchecked<D, T, Op>(lhs: Tensor<D, T>, rhs: Tensor<D, T>) -> Tensor<D, T>
where
    D: Device + Clone,
    T: Scalar,
    Op: From<InnerOp<2, 1>> + TensorOp,
{
    let mut output = Tensor::zeros_like(&lhs);
    let mut ops = [lhs.tape().ops.clone(), rhs.tape().ops.clone()]
        .concat()
        .into_iter()
        .unique()
        .collect_vec();

    let inputs = [lhs.ir(Access::ReadOnly), rhs.ir(Access::ReadOnly)];
    let outputs = [output.ir(Access::WriteOnly)];
    let op = Op::from(InnerOp::new(inputs, outputs));
    ops.push(Box::new(op));
    output.replace_ops(ops);

    output
}

impl<D: Device + Clone, T: Scalar> Tensor<D, T> {
    /// Replace the inner ops recorded on the tape of the tensor.
    fn replace_ops(&mut self, ops: Vec<Box<dyn TensorOp>>) -> Vec<Box<dyn TensorOp>> {
        let tape = self.tape_mut();
        std::mem::replace(&mut tape.ops, ops)
    }

    /// Create a new tensor with the given device, layout, and contents.
    pub fn create<L, C>(device: D, layout: L, contents: C) -> Self
    where
        L: IntoLayout,
        C: Into<Arc<[T]>>,
    {
        let layout = layout.into_layout();
        let contents: Arc<[T]> = contents.into();
        let size = contents.len() * size_of::<T>();

        let mut output = Tensor::<D, T>::zeros(device, layout, size);
        let op = InnerOp::new([], [output.ir(Access::WriteOnly)]);
        let op = CreateOp { op, contents };
        let ops: Vec<Box<dyn TensorOp>> = vec![Box::new(op)];
        output.replace_ops(ops);

        output
    }

    /// Read back the contents of the tensor from the device.
    #[inline]
    pub async fn back(self) -> Result<Box<[T]>, DeviceError> {
        let (sender, receiver) = flume::bounded(0);
        let tape = self.tape().clone();
        let event = DeviceEvent::Back { tape, sender };
        self.device().execute(event);

        // reclaim self using a terminal op
        let mut tape = self.tape().clone();
        let op = OneOp::new(self.ir(Access::ReadOnly));
        tape.ops.push(Box::new(op));

        let (sender, _) = flume::bounded(0);
        let event = DeviceEvent::Execute { tape, sender };
        self.device().execute(event);

        let data = receiver.recv_async().await??;
        Ok(bytemuck::cast_slice(&data.0).to_vec().into_boxed_slice())
    }
}

impl<D: Device + Clone, T: Scalar> std::ops::Add<Tensor<D, T>> for Tensor<D, T> {
    type Output = Tensor<D, T>;

    fn add(self, rhs: Tensor<D, T>) -> Self::Output {
        assert_eq!(self.layout(), rhs.layout(), "tensor layouts must match");
        binary_op_unchecked::<_, _, AddOp<T>>(self, rhs)
    }
}

impl<D: Device + Clone, T: Float> Tensor<D, T> {
    #[inline]
    pub fn softmax(self) -> Self {
        let mut output = Tensor::<D, T>::zeros_like(&self);
        let mut ops = self.tape().ops.clone();

        let phantom = PhantomData;
        let op = InnerOp::new([self.ir(Access::ReadOnly)], [output.ir(Access::WriteOnly)]);
        let op = SoftmaxOp::<T> { op, phantom };
        ops.push(Box::new(op));

        output.replace_ops(ops);
        output
    }

    #[inline]
    pub fn layer_norm(self, w: Tensor<D, f16>, b: Tensor<D, f16>, eps: f32) -> Self {
        let layout = self.layout();
        let shape = [layout.shape_of(0)].into();
        assert_eq!(w.layout().shape(), shape, "weight must match input shape");
        assert_eq!(b.layout().shape(), shape, "bias must match input shape");

        let mut output = Tensor::<D, T>::zeros_like(&self);
        let mut ops = [
            self.tape().ops.clone(),
            w.tape().ops.clone(),
            b.tape().ops.clone(),
        ]
        .concat()
        .into_iter()
        .unique()
        .collect_vec();

        let phantom = PhantomData;
        let inputs = [
            self.ir(Access::ReadOnly),
            w.ir(Access::ReadOnly),
            b.ir(Access::ReadOnly),
        ];
        let outputs = [output.ir(Access::WriteOnly)];
        let op = InnerOp::new(inputs, outputs);
        let op = LayerNormOp::<T> { op, eps, phantom };
        ops.push(Box::new(op));

        output.replace_ops(ops);
        output
    }
}
