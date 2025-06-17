use std::{marker::PhantomData, sync::Arc};

use half::f16;
use mia_derive::build_api;

use super::ops::{AddOp, CreateOp, LayerNormOp, SoftmaxOp};
use crate::loom::{
    device::{Device, DeviceError, DeviceEvent},
    layout::IntoLayout,
    num::{Float, Scalar},
    ops::{Access, InnerOp, OneOp, TensorOp},
    tensor::Tensor,
};

build_api!(1);
build_api!(2);
build_api!(3);

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
        let phantom = PhantomData;
        build_api_2(move |op| AddOp::<T> { op, phantom }, self, rhs)
    }
}

impl<D: Device + Clone, T: Float> Tensor<D, T> {
    #[inline]
    pub fn softmax(self) -> Self {
        let phantom = PhantomData;
        build_api_1(move |op| SoftmaxOp::<T> { op, phantom }, self)
    }

    #[inline]
    pub fn layer_norm(self, w: Tensor<D, f16>, b: Tensor<D, f16>, eps: f32) -> Self {
        let layout = self.layout();
        let shape = [layout.shape_of(0)].into();
        assert_eq!(w.layout().shape(), shape, "weight must match input shape");
        assert_eq!(b.layout().shape(), shape, "bias must match input shape");

        let phantom = PhantomData;
        build_api_3(move |op| LayerNormOp::<T> { op, eps, phantom }, self, w, b)
    }
}
