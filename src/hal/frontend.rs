use std::{marker::PhantomData, sync::Arc};

use derive_more::{Deref, DerefMut};
use half::f16;
use mia_derive::build_api;

use super::ops::{AddOp, CreateOp, L2NormOp, LayerNormOp, MatMatFp16Op, MulOp, SoftmaxOp};
use crate::loom::{
    device::{Device, DeviceError, DeviceEvent},
    layout::{IntoLayout, Layout},
    num::{F16x4, Float, Scalar},
    ops::{Access, InnerOp, Mermaid, OneOp, TensorOp},
    tensor::{Tensor, TensorError},
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
    pub fn create<L, C>(device: D, layout: L, contents: C) -> Result<Self, TensorError>
    where
        L: IntoLayout,
        C: Into<Arc<[T]>>,
    {
        let layout = layout.into_layout();
        let contents: Arc<[T]> = contents.into();
        let size = size_of_val(&contents[..]);

        let mut output = Tensor::<D, T>::init(device, layout, size)?;
        let op = InnerOp::new([], [output.ir(Access::WriteOnly)]);
        let op = CreateOp { op, contents };
        let ops: Vec<Box<dyn TensorOp>> = vec![Box::new(op)];
        output.replace_ops(ops);

        Ok(output)
    }

    /// Generate a Mermaid diagram representation of the tensor's computation graph.
    ///
    /// This function executes the tensor's computation graph on the device and returns a `Mermaid` struct,
    /// which can be used to visualize the graph as a Mermaid diagram.
    #[inline]
    pub async fn mermaid(self) -> Result<Mermaid, DeviceError> {
        let (sender, receiver) = flume::bounded(0);
        let tape = self.tape().clone();
        let event = DeviceEvent::Execute { tape, sender };
        self.device().execute(event);
        let data = receiver.recv_async().await??;
        Ok(data.0)
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
        self.try_add(rhs).expect("tensor layouts must match")
    }
}

impl<D: Device + Clone, T: Scalar> std::ops::Mul<Tensor<D, T>> for Tensor<D, T> {
    type Output = Tensor<D, T>;

    fn mul(self, rhs: Tensor<D, T>) -> Self::Output {
        self.try_mul(rhs).expect("tensor layouts must match")
    }
}

impl<D: Device + Clone, T: Scalar> Tensor<D, T> {
    /// # Element-wise Addition (`try_add`)
    /// Performs element-wise addition between two tensors.
    ///
    /// ## Arguments
    /// * `self` - The first tensor operand
    /// * `rhs` - The second tensor operand, must have the same layout as `self`
    ///
    /// ## Returns
    /// * `Result<Self, TensorError>` - A new tensor containing the element-wise sum,
    ///   or an error if the tensor layouts don't match
    #[inline]
    pub fn try_add(self, rhs: Tensor<D, T>) -> Result<Self, TensorError> {
        let layout = self.layout();
        let rhs = rhs.check_layout(layout)?;

        let phantom = PhantomData::<T>;
        let f = move |op| AddOp { op, phantom };
        let output = Tensor::zeros_like(&self);
        Ok(build_api_2(f, output, self, rhs))
    }

    /// # Element-wise Multiplication (`try_mul`)
    /// Performs element-wise multiplication between two tensors.
    ///
    /// ## Arguments
    /// * `self` - The first tensor operand
    /// * `rhs` - The second tensor operand, must have the same layout as `self`
    ///
    /// ## Returns
    /// * `Result<Self, TensorError>` - A new tensor containing the element-wise product,
    ///   or an error if the tensor layouts don't match
    #[inline]
    pub fn try_mul(self, rhs: Tensor<D, T>) -> Result<Self, TensorError> {
        let layout = self.layout();
        let rhs = rhs.check_layout(layout)?;

        let phantom = PhantomData::<T>;
        let f = move |op| MulOp { op, phantom };
        let output = Tensor::zeros_like(&self);
        Ok(build_api_2(f, output, self, rhs))
    }
}

impl<D: Device + Clone, T: Float> Tensor<D, T> {
    /// # Softmax (`softmax`)
    /// Performs softmax normalization on the input tensor.
    ///
    /// ## Arguments
    /// * `self` - The input tensor to normalize, of shape `[M*]`.
    ///
    /// ## Returns
    /// * `Tensor<D, T>` - A new tensor with the same shape as the input.
    #[inline]
    pub fn softmax(self) -> Self {
        let phantom = PhantomData;
        let f = move |op| SoftmaxOp::<T> { op, phantom };
        let output = Tensor::zeros_like(&self);
        build_api_1(f, output, self)
    }

    /// # Layer Normalization (`layer_norm`)
    /// Performs layer normalization on the input tensor using the given weight and bias tensors.
    ///
    /// ## Arguments
    /// * `self` - The input tensor to normalize, of shape `[M, N*]`.
    /// * `w` - The weight tensor of shape `[M]`.
    /// * `b` - The bias tensor of shape `[M]`.
    /// * `eps` - A small value added to the variance to avoid division by zero.
    ///
    /// ## Returns
    /// * `Result<Tensor<D, T>, TensorError>` - A new tensor with the same shape as the input,
    ///   or an error if the dimensions are incompatible.
    #[inline]
    pub fn layer_norm(
        self,
        w: Tensor<D, f16>,
        b: Tensor<D, f16>,
        eps: f32,
    ) -> Result<Self, TensorError> {
        let [m] = self.layout().shape().to_array();
        let w = w.check_dim(1..=1)?.check_shape(m)?;
        let b = b.check_dim(1..=1)?.check_shape(m)?;

        let phantom = PhantomData::<T>;
        let f = |op| LayerNormOp { op, eps, phantom };
        let output = Tensor::zeros_like(&self);
        Ok(build_api_3(f, output, self, w, b))
    }

    /// # L2 Normalization (`l2_norm`)
    /// Performs L2 normalization on the input tensor, normalizing each vector to unit length.
    ///
    /// ## Arguments
    /// * `self` - The input tensor to normalize.
    /// * `eps` - A small value added to the norm to avoid division by zero.
    ///
    /// ## Returns
    /// * `Self` - A new tensor with the same shape as the input, where each vector is normalized to unit length.
    #[inline]
    pub fn l2_norm(self, eps: f32) -> Self {
        let phantom = PhantomData::<T>;
        let f = |op| L2NormOp { op, eps, phantom };
        let output = Tensor::zeros_like(&self);
        build_api_1(f, output, self)
    }
}

#[derive(Debug, Clone, Deref, DerefMut)]
pub struct MatrixFp16<D>(pub Tensor<D, F16x4>);

impl<D: Device + Clone> MatrixFp16<D> {
    #[inline]
    pub fn from_tensor(tensor: Tensor<D, f16>) -> Result<Self, TensorError> {
        let [k, m] = tensor.layout().pad_to(2).shape().try_to_array()?;
        let tensor = tensor.check_layout([k, m])?;
        let tensor = tensor.cast([f16::index::<F16x4>(k), m])?;
        Ok(Self(tensor))
    }

    /// # Matrix Multiplication (`matmul`)
    /// Performs batched matrix multiplication between this `MatrixFp16` and another tensor.
    ///
    /// ## Arguments
    /// * `self` - A `MatrixFp16` of shape `[K, M]`.
    /// * `rhs` - The right-hand side tensor of shape `[K, N]` (transposed).
    ///
    /// The inner dimension `K` must match between `self` and `rhs`.
    ///
    /// ## Returns
    /// * `Result<Tensor<D, T>, TensorError>` - A new tensor of shape `[M, N]` containing the result,
    ///   or an error if the dimensions are incompatible.
    #[inline]
    pub fn matmul<T: Float>(self, rhs: Tensor<D, T>) -> Result<Tensor<D, T>, TensorError> {
        let [k, n] = rhs.layout().shape().try_to_array()?;
        let rhs = rhs.cast::<T::Float4>([T::index::<T::Float4>(k), n])?;

        let [k, m] = self.layout().shape().try_to_array()?;

        let lhs = self.0.check_layout([k, m])?;
        let rhs = rhs.check_layout([k, n])?;

        let (bk, bm, bn) = (4, 16, 16);

        let layouts = [
            Layout::from_shape([k, m]),
            Layout::from_shape([k, n]),
            Layout::from_shape([m, n]),
        ];
        let layouts = [
            layouts[0].div_tiler([(bk, 1), (bm, 1)])?,
            layouts[1].div_tiler([(bk, 1), (bn, 1)])?,
            layouts[2].div_tiler([(bm, 1), (bn, 1)])?,
        ];

        let phantom = PhantomData::<T>;
        let f = |op| MatMatFp16Op {
            op,
            phantom,
            layouts,
        };
        let device = lhs.device().clone();
        let output = Tensor::zeros(device, [m, n]);
        Ok(build_api_2(f, output, lhs, rhs))
    }
}
