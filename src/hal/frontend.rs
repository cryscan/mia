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

fn build_api_1<D, U, Op, F>(t: Tensor<D, impl Scalar>, f: F) -> Tensor<D, U>
where
    D: Device + Clone,
    U: Scalar,
    Op: TensorOp,
    F: FnOnce(InnerOp<1, 1>) -> Op,
{
    let mut output = Tensor::zeros_like(&t);
    let mut ops = t.tape().ops.clone();

    let inputs = [t.ir(Access::ReadOnly)];
    let outputs = [output.ir(Access::WriteOnly)];
    let op = f(InnerOp::new(inputs, outputs));
    ops.push(Box::new(op));
    output.replace_ops(ops);

    output
}

fn build_api_2<D, U, Op, F>(
    t0: Tensor<D, impl Scalar>,
    t1: Tensor<D, impl Scalar>,
    f: F,
) -> Tensor<D, U>
where
    D: Device + Clone,
    U: Scalar,
    Op: TensorOp,
    F: FnOnce(InnerOp<2, 1>) -> Op,
{
    let mut output = Tensor::zeros_like(&t0);
    let mut ops = [t0.tape().ops.clone(), t1.tape().ops.clone()]
        .concat()
        .into_iter()
        .unique()
        .collect_vec();

    let inputs = [t0.ir(Access::ReadOnly), t1.ir(Access::ReadOnly)];
    let outputs = [output.ir(Access::WriteOnly)];
    let op = f(InnerOp::new(inputs, outputs));
    ops.push(Box::new(op));
    output.replace_ops(ops);

    output
}

fn build_api_3<D, U, Op, F>(
    t0: Tensor<D, impl Scalar>,
    t1: Tensor<D, impl Scalar>,
    t2: Tensor<D, impl Scalar>,
    f: F,
) -> Tensor<D, U>
where
    D: Device + Clone,
    U: Scalar,
    Op: TensorOp,
    F: FnOnce(InnerOp<3, 1>) -> Op,
{
    let mut output = Tensor::zeros_like(&t0);
    let mut ops = [
        t0.tape().ops.clone(),
        t1.tape().ops.clone(),
        t2.tape().ops.clone(),
    ]
    .concat()
    .into_iter()
    .unique()
    .collect_vec();

    let inputs = [
        t0.ir(Access::ReadOnly),
        t1.ir(Access::ReadOnly),
        t2.ir(Access::ReadOnly),
    ];
    let outputs = [output.ir(Access::WriteOnly)];
    let op = f(InnerOp::new(inputs, outputs));
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
        let phantom = PhantomData;
        build_api_2(self, rhs, move |op| AddOp::<T> { op, phantom })
    }
}

impl<D: Device + Clone, T: Float> Tensor<D, T> {
    #[inline]
    pub fn softmax(self) -> Self {
        let phantom = PhantomData;
        build_api_1(self, move |op| SoftmaxOp::<T> { op, phantom })
    }

    #[inline]
    pub fn layer_norm(self, w: Tensor<D, f16>, b: Tensor<D, f16>, eps: f32) -> Self {
        let layout = self.layout();
        let shape = [layout.shape_of(0)].into();
        assert_eq!(w.layout().shape(), shape, "weight must match input shape");
        assert_eq!(b.layout().shape(), shape, "bias must match input shape");

        let phantom = PhantomData;
        build_api_3(self, w, b, move |op| LayerNormOp::<T> { op, eps, phantom })
    }
}
