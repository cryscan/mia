use std::{borrow::Cow, cell::RefCell, collections::VecDeque, rc::Rc};

use itertools::Itertools;
use rustc_hash::FxHashMap as HashMap;
use thiserror::Error;

use super::{Backend, BackendOp};
use crate::loom::{
    ops::{Access, TensorIr, TensorOp},
    tensor::TensorId,
};

#[derive(Debug, Error)]
pub enum AllocError {
    #[error("violation of write uniqueness rule: {0}")]
    WriteOnly(TensorId),
    #[error("violation of read/write uniqueness rule: {0}")]
    ReadWrite(TensorId),
}

#[derive(Debug, Default, Clone)]
pub struct Allocator {
    alloc: HashMap<TensorId, TensorId>,
    free: HashMap<usize, VecDeque<TensorId>>,
}

impl Allocator {
    /// Redirects `id` to its root and folds path. Returns the root `id`.
    pub fn redirect(&mut self, id: TensorId) -> TensorId {
        let mut root = id;
        while let Some(&parent) = self.alloc.get(&root) {
            assert_ne!(parent, id);
            root = parent;
        }
        if root != id {
            self.alloc.insert(id, root);
        }
        root
    }

    /// Checks if mutation uniqueness rules applies.
    fn check(&mut self, io: &[TensorIr]) -> Result<(), AllocError> {
        macro_rules! return_eq {
            ($x:expr, $y:expr, $ret:expr) => {
                if $x == $y {
                    return $ret;
                }
            };
        }
        for (x, y) in io.iter().tuple_combinations() {
            // 1. `WriteOnly` ids must be unique
            if matches!(x.access, Access::WriteOnly) || matches!(y.access, Access::WriteOnly) {
                let x = self.redirect(x.id);
                let y = self.redirect(y.id);
                return_eq!(x, y, Err(AllocError::WriteOnly(x)))
            }
            // 2. `ReadWrite` ids mut be unique unless the other is also `ReadWrite`
            if matches!(x.access, Access::ReadWrite) ^ matches!(y.access, Access::ReadWrite) {
                let x = self.redirect(x.id);
                let y = self.redirect(y.id);
                return_eq!(x, y, Err(AllocError::ReadWrite(x)))
            }
        }
        Ok(())
    }

    pub fn alloc(&mut self, op: Box<dyn TensorOp>) -> Result<AllocOp, AllocError> {
        let io = op.io();
        self.check(&io)?;

        // 0. convert to shared ref with internal mutability
        let io = io.into_iter().map(RefCell::new).map(Rc::new).collect_vec();

        // 1. substitute tensor ids following the redirection map
        for ir in io.iter() {
            let id = ir.borrow().id;
            ir.borrow_mut().id = self.redirect(id);
        }

        // 2. tensors that can be immediately reused
        let mut free = io
            .iter()
            .filter(|ir| matches!(ir.borrow().access, Access::ReadOnly))
            .filter(|ir| ir.borrow().count <= 1)
            .cloned()
            .collect_vec();

        // 3. reuse tensors from the local free list
        for x in io
            .iter()
            .filter(|ir| matches!(ir.borrow().access, Access::WriteOnly))
        {
            if let Some((index, _)) = free
                .iter()
                .find_position(|y| y.borrow().is_compatible(&x.borrow()))
            {
                let y = free.remove(index);
                x.borrow_mut().access = Access::ReadWrite;
                y.borrow_mut().access = Access::ReadWrite;
                self.alloc.insert(x.borrow().id, y.borrow().id);
            }
        }

        // 4. reuse tensors from the allocator's free list
        for ir in io
            .iter()
            .filter(|ir| matches!(ir.borrow().access, Access::WriteOnly))
        {
            let size = &ir.borrow().data_size();
            if let Some(id) = self.free.get_mut(size).and_then(|free| free.pop_front()) {
                self.alloc.insert(ir.borrow().id, id);
            }
        }

        // 5. substitute tensor ids once more after updating the map
        for ir in io.iter() {
            let id = ir.borrow().id;
            ir.borrow_mut().id = self.redirect(id);
        }

        // 6. update the free list
        for ir in free {
            let size = ir.borrow().data_size();
            let id = ir.borrow().id;
            let mut ids = self.free.remove(&size).unwrap_or_default();
            ids.push_back(id);
            self.free.insert(size, ids);
        }

        let io = io
            .into_iter()
            .map(|ir| Rc::into_inner(ir).unwrap().into_inner())
            .collect_vec();

        // check for reused free ids
        self.check(&io)?;

        Ok(AllocOp { op, io })
    }

    #[inline]
    pub fn reset(&mut self) {
        *self = Default::default()
    }
}

/// A wrapper around another [`TensorOp`], of which storages are optimized by the allocator.
#[derive(Debug, Clone)]
pub struct AllocOp {
    op: Box<dyn TensorOp>,
    io: Vec<TensorIr>,
}

impl TensorOp for AllocOp {
    #[inline]
    fn name(&self) -> Cow<'static, str> {
        let type_name = std::any::type_name::<Self>();
        let op_name = self.op.name();
        Cow::from(format!("{type_name}<{op_name}>"))
    }

    #[inline]
    fn io(&self) -> Vec<TensorIr> {
        self.io.clone()
    }
}

impl<B: Backend> BackendOp<AllocOp> for B {
    #[inline]
    fn execute(&self, op: AllocOp, io: Vec<TensorIr>) {
        self.execute(op.op, io);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod tests {
    use std::{borrow::Cow, time::Duration};

    use anyhow::Result;
    use itertools::Itertools;

    use crate::loom::{
        device::{BackendOp, CpuBuilder, Device, cpu},
        layout::Layout,
        num::DataType,
        ops::{Access, TensorIr, TensorOp, TensorTape},
        tensor::TensorId,
    };

    #[derive(Debug, Clone)]
    struct PhonyBinaryOp {
        input: [TensorIr; 2],
        output: TensorIr,
    }

    impl TensorOp for PhonyBinaryOp {
        fn name(&self) -> Cow<'static, str> {
            Cow::Borrowed("BinaryOp")
        }

        fn io(&self) -> Vec<TensorIr> {
            vec![
                self.input[0].clone(),
                self.input[1].clone(),
                self.output.clone(),
            ]
        }
    }

    impl BackendOp<PhonyBinaryOp> for cpu::Backend {
        fn execute(&self, op: PhonyBinaryOp, io: Vec<TensorIr>) {
            println!("{}", op.name());
            for (x, y) in op.io().into_iter().zip_eq(io) {
                println!("{}\t{}\t→\t{}\t{}", x.access, x.id, y.access, y.id);
            }
            println!()
        }
    }

    #[derive(Debug, Clone)]
    struct PhonyUnaryOp {
        input: TensorIr,
        output: TensorIr,
    }

    impl TensorOp for PhonyUnaryOp {
        fn name(&self) -> Cow<'static, str> {
            Cow::Borrowed("UnaryOp")
        }

        fn io(&self) -> Vec<TensorIr> {
            vec![self.input.clone(), self.output.clone()]
        }
    }

    impl BackendOp<PhonyUnaryOp> for cpu::Backend {
        fn execute(&self, op: PhonyUnaryOp, io: Vec<TensorIr>) {
            println!("{}", op.name());
            for (x, y) in op.io().into_iter().zip_eq(io) {
                println!("{}\t{}\t→\t{}\t{}", x.access, x.id, y.access, y.id);
            }
            println!()
        }
    }

    const ID_MAP: [TensorId; 8] = [
        TensorId(uuid::uuid!("00000000-0000-0000-0000-ffff00000000")),
        TensorId(uuid::uuid!("00000000-0000-0000-0000-ffff00000001")),
        TensorId(uuid::uuid!("00000000-0000-0000-0000-ffff00000002")),
        TensorId(uuid::uuid!("00000000-0000-0000-0000-ffff00000003")),
        TensorId(uuid::uuid!("00000000-0000-0000-0000-ffff00000004")),
        TensorId(uuid::uuid!("00000000-0000-0000-0000-ffff00000005")),
        TensorId(uuid::uuid!("00000000-0000-0000-0000-ffff00000006")),
        TensorId(uuid::uuid!("00000000-0000-0000-0000-ffff00000007")),
    ];

    #[tokio::test]
    async fn test_reuse() -> Result<()> {
        let cpu = CpuBuilder::new()
            .add_op::<PhonyBinaryOp>()
            .add_op::<PhonyUnaryOp>()
            .build()
            .await;

        let ops: Vec<Box<dyn TensorOp>> = vec![
            Box::new(PhonyBinaryOp {
                input: [
                    TensorIr {
                        layout: Layout::from_shape([32]),
                        r#type: DataType::F32x4,
                        id: ID_MAP[0],
                        count: 1,
                        access: Access::ReadOnly,
                    },
                    TensorIr {
                        layout: Layout::from_shape([32]),
                        r#type: DataType::F16x4,
                        id: ID_MAP[1],
                        count: 1,
                        access: Access::ReadOnly,
                    },
                ],
                output: TensorIr {
                    layout: Layout::from_shape([32]),
                    r#type: DataType::F16x4,
                    id: ID_MAP[2],
                    count: 1,
                    access: Access::WriteOnly,
                },
            }),
            Box::new(PhonyBinaryOp {
                input: [
                    TensorIr {
                        layout: Layout::from_shape([32]),
                        r#type: DataType::F32x4,
                        id: ID_MAP[2],
                        count: 2,
                        access: Access::ReadOnly,
                    },
                    TensorIr {
                        layout: Layout::from_shape([32]),
                        r#type: DataType::F16x4,
                        id: ID_MAP[3],
                        count: 1,
                        access: Access::ReadOnly,
                    },
                ],
                output: TensorIr {
                    layout: Layout::from_shape([32]),
                    r#type: DataType::F32x4,
                    id: ID_MAP[4],
                    count: 1,
                    access: Access::WriteOnly,
                },
            }),
            Box::new(PhonyBinaryOp {
                input: [
                    TensorIr {
                        layout: Layout::from_shape([32]),
                        r#type: DataType::F32x4,
                        id: ID_MAP[4],
                        count: 1,
                        access: Access::ReadOnly,
                    },
                    TensorIr {
                        layout: Layout::from_shape([32]),
                        r#type: DataType::F32x4,
                        id: ID_MAP[2],
                        count: 2,
                        access: Access::ReadOnly,
                    },
                ],
                output: TensorIr {
                    layout: Layout::from_shape([32]),
                    r#type: DataType::F32x4,
                    id: ID_MAP[5],
                    count: 1,
                    access: Access::WriteOnly,
                },
            }),
            Box::new(PhonyUnaryOp {
                input: TensorIr {
                    layout: Layout::from_shape([32]),
                    r#type: DataType::F32x4,
                    id: ID_MAP[2],
                    count: 1,
                    access: Access::ReadOnly,
                },
                output: TensorIr {
                    layout: Layout::from_shape([32]),
                    r#type: DataType::F32x4,
                    id: ID_MAP[6],
                    count: 1,
                    access: Access::WriteOnly,
                },
            }),
        ];
        let this = ID_MAP[5];
        let tape = TensorTape { this, ops };
        cpu.execute(tape);

        tokio::time::sleep(Duration::from_secs(1)).await;
        Ok(())
    }
}
