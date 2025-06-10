use std::sync::Arc;

use itertools::Itertools;

use crate::{
    cpu::backend::{AddOp, BinaryOp},
    loom::{device::Cpu, num::Scalar, ops::Access, tensor::Tensor},
};

impl<T> std::ops::Add<Tensor<Cpu, T>> for Tensor<Cpu, T>
where
    T: Scalar + std::ops::Add<Output = T>,
{
    type Output = Tensor<Cpu, T>;

    fn add(self, rhs: Tensor<Cpu, T>) -> Self::Output {
        assert_eq!(self.layout(), rhs.layout());

        let mut output = Self::zeros_like(&self);
        let mut ops = [self.as_ref().ops.clone(), rhs.as_ref().ops.clone()]
            .concat()
            .into_iter()
            .unique()
            .collect_vec();

        let op = AddOp(BinaryOp {
            id: Default::default(),
            inputs: [self.ir(Access::ReadOnly), rhs.ir(Access::ReadOnly)],
            output: output.ir(Access::WriteOnly),
        });
        ops.push(Box::new(op));

        let tape = output.as_mut();
        let tape = Arc::get_mut(tape).expect("must be unique");
        tape.ops = ops;

        output
    }
}
