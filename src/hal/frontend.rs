use std::sync::Arc;

use itertools::Itertools;

use crate::{
    hal::ops::{AddOp, BinaryOp},
    loom::{
        device::Device,
        num::Scalar,
        ops::{Access, TensorOp},
        tensor::Tensor,
    },
};

fn element_binary_op<D, T, Op>(lhs: Tensor<D, T>, rhs: Tensor<D, T>) -> Tensor<D, T>
where
    D: Device,
    T: Scalar,
    Op: From<BinaryOp> + TensorOp,
{
    assert_eq!(lhs.layout(), rhs.layout());

    let mut output = Tensor::zeros_like(&lhs);
    let mut ops = [lhs.as_ref().ops.clone(), rhs.as_ref().ops.clone()]
        .concat()
        .into_iter()
        .unique()
        .collect_vec();

    let op = Op::from(BinaryOp {
        id: Default::default(),
        inputs: [lhs.ir(Access::ReadOnly), rhs.ir(Access::ReadOnly)],
        output: output.ir(Access::WriteOnly),
    });
    ops.push(Box::new(op));

    let tape = output.as_mut();
    let tape = Arc::get_mut(tape).expect("must be unique");
    tape.ops = ops;

    output
}

impl<D, T> std::ops::Add<Tensor<D, T>> for Tensor<D, T>
where
    D: Device,
    T: Scalar,
{
    type Output = Tensor<D, T>;

    fn add(self, rhs: Tensor<D, T>) -> Self::Output {
        element_binary_op::<_, _, AddOp<T>>(self, rhs)
    }
}
