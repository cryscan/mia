use std::{borrow::Cow, cell::RefCell, collections::VecDeque};

use derive_more::{Deref, DerefMut, Display};
use itertools::Itertools;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use thiserror::Error;

use super::Backend;
use crate::loom::{
    ops::{Access, BackendOp, Mermaid, TensorIr, TensorOp, TensorOpId, TensorTape},
    tensor::TensorId,
};

#[derive(Debug, Default, Display, Clone, Copy, PartialEq, Eq, Hash, Deref, DerefMut)]
pub struct StashId(uid::Id<StashId>);

#[derive(Debug, Error)]
pub enum AllocError {
    #[error("violation of write uniqueness rule: {0}")]
    WriteOnly(TensorId),
    #[error("violation of read/write uniqueness rule: {0}")]
    ReadWrite(TensorId),
}

#[derive(Debug, Default, Clone)]
pub struct Allocator {
    stash: HashMap<TensorId, StashId>,
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

    /// Retrieve a tensor's allocated stash location.
    /// Allocates a new stash location if needed.
    pub fn retrieve(&mut self, id: TensorId) -> StashId {
        let root = self.redirect(id);
        match self.stash.get(&root) {
            Some(&id) => id,
            None => {
                let id = StashId(uid::Id::new());
                self.stash.insert(root, id);
                id
            }
        }
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

    /// Allocates actual locations for tensors in the op and returns a wrapped [`AllocOp`].
    pub fn alloc(&mut self, op: Box<dyn TensorOp>) -> Result<AllocOp, AllocError> {
        let io = op.io();
        self.check(&io)?;

        // 0. convert to shared ref with internal mutability
        let io = io.into_iter().map(RefCell::new).collect_vec();

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

        // 6. release freed ids and update the free list
        for ir in free {
            let id = ir.borrow().id;
            let size = ir.borrow().data_size();

            self.alloc.retain(|_, &mut x| x != id);

            let mut ids = self.free.remove(&size).unwrap_or_default();
            ids.push_back(id);
            self.free.insert(size, ids);
        }

        let io = io.into_iter().map(|ir| ir.into_inner()).collect_vec();

        // check for reused free ids
        self.check(&io)?;

        Ok(AllocOp { op, io })
    }

    /// Retains only the specified tensor IDs and their dependencies in the allocator.
    pub fn retain(&mut self, retain: Vec<TensorId>) {
        let ids = retain.iter().map(|&id| self.redirect(id));
        let ids: HashSet<_> = retain.iter().copied().chain(ids).collect();

        self.stash.retain(|id, _| ids.contains(id));
        self.alloc.retain(|k, v| ids.contains(k) || ids.contains(v));
        self.free
            .values_mut()
            .for_each(|v| v.retain(|id| ids.contains(id)));
    }

    /// Prints the allocator's state in a human-readable format.
    pub fn print_pretty(&self) -> String {
        let ids: HashSet<_> = self
            .alloc
            .keys()
            .chain(self.alloc.values())
            .copied()
            .collect();
        ids.into_iter()
            .map(|id| {
                let key = self.alloc.get(&id).unwrap_or(&id);
                let stash = self
                    .stash
                    .get(key)
                    .copied()
                    .map_or(Default::default(), |id| id.to_string());
                format!("{id}\t→ {key}\t: {stash}")
            })
            .join("\n")
    }
}

impl std::fmt::Display for Allocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.print_pretty())
    }
}

impl TensorTape {
    pub fn mermaid_alloc<A>(&self, mut allocator: A) -> Mermaid
    where
        A: std::ops::DerefMut<Target = Allocator>,
    {
        let allocator = allocator.deref_mut();
        let mut s = "graph TD\n".to_string();

        for (index, op) in self.ops.iter().enumerate() {
            let op_name = op.name();
            let op_node = format!("op_{index}");
            let op_label = format!("{op_name}");
            s.push_str(&format!("    {op_node}[\"{op_label}\"]\n"));

            for ir in op.io() {
                let tensor_node = format!("tensor_{}", ir.id);
                let tensor_label = format!("{}", allocator.retrieve(ir.id));

                match ir.access {
                    Access::ReadOnly => {
                        s.push_str(&format!("    {tensor_node}((\"{tensor_label}\"))\n"));
                        s.push_str(&format!("    {tensor_node} --> |Read| {op_node}\n"));
                    }
                    Access::WriteOnly => {
                        s.push_str(&format!("    {tensor_node}((\"{tensor_label}\"))\n"));
                        s.push_str(&format!("    {op_node} --> |Write| {tensor_node}\n"));
                    }
                    _ => {}
                }
            }
        }
        Mermaid(s)
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
    fn id(&self) -> TensorOpId {
        self.op.id()
    }

    #[inline]
    fn io(&self) -> Vec<TensorIr> {
        self.io.clone()
    }
}

impl<B: Backend> BackendOp<B> for AllocOp {
    async fn execute(&self, backend: &mut B, io: Vec<TensorIr>) {
        backend.execute(self.op.as_ref(), io).await;
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use itertools::Itertools;

    use super::Allocator;
    use crate::loom::{
        device::{CpuBuilder, Device, DeviceEvent, cpu},
        layout::Layout,
        num::DataType,
        ops::{Access, BackendOp, TensorIr, TensorOp, TensorOpId, TensorTape},
        slice::Slice,
        tensor::TensorId,
    };

    fn check_ir(allocator: &mut Allocator, x: TensorIr, y: TensorIr) {
        println!(
            "{:<12}{:<8}{:<4}{}\t→\t{:<12}{:<8}{}\t{}",
            format!("{}", x.access),
            format!("{}", x.r#type),
            x.count,
            x.id,
            format!("{}", y.access),
            format!("{}", y.r#type),
            y.id,
            allocator.retrieve(x.id),
        );

        // 1. data sizes must match
        assert_eq!(x.data_size(), y.data_size());

        // 2. types must match in the case of `ReadWrite`
        if matches!(y.access, Access::ReadWrite) {
            assert_eq!(x.r#type, y.r#type);
        }
    }

    #[derive(Debug, Clone)]
    struct PhonyBinaryOp {
        id: TensorOpId,
        input: [TensorIr; 2],
        output: TensorIr,
    }

    impl TensorOp for PhonyBinaryOp {
        fn id(&self) -> TensorOpId {
            self.id
        }

        fn io(&self) -> Vec<TensorIr> {
            vec![
                self.input[0].clone(),
                self.input[1].clone(),
                self.output.clone(),
            ]
        }
    }

    impl BackendOp<cpu::Backend> for PhonyBinaryOp {
        async fn execute(&self, backend: &mut cpu::Backend, io: Vec<TensorIr>) {
            println!("{}", self.name());
            self.io()
                .into_iter()
                .zip_eq(io)
                .for_each(|(x, y)| check_ir(&mut backend.allocator(), x, y));
            println!();
        }
    }

    #[derive(Debug, Clone)]
    struct PhonyUnaryOp {
        id: TensorOpId,
        input: TensorIr,
        output: TensorIr,
    }

    impl TensorOp for PhonyUnaryOp {
        fn id(&self) -> TensorOpId {
            self.id
        }

        fn io(&self) -> Vec<TensorIr> {
            vec![self.input.clone(), self.output.clone()]
        }
    }

    impl BackendOp<cpu::Backend> for PhonyUnaryOp {
        async fn execute(&self, backend: &mut cpu::Backend, io: Vec<TensorIr>) {
            println!("{}", self.name());
            self.io()
                .into_iter()
                .zip_eq(io)
                .for_each(|(x, y)| check_ir(&mut backend.allocator(), x, y));
            println!();
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
    async fn test_reuse() -> Result<(), Box<dyn Error>> {
        let cpu = CpuBuilder::new()
            .add_op::<PhonyBinaryOp>()
            .add_op::<PhonyUnaryOp>()
            .build()
            .await;

        let ops: Vec<Box<dyn TensorOp>> = vec![
            Box::new(PhonyBinaryOp {
                id: Default::default(),
                input: [
                    TensorIr {
                        layout: Layout::from_shape([32]),
                        slice: Slice::from(..),
                        r#type: DataType::F32x4,
                        id: ID_MAP[0],
                        count: 1,
                        access: Access::ReadOnly,
                    },
                    TensorIr {
                        layout: Layout::from_shape([32]),
                        slice: Slice::from(..),
                        r#type: DataType::F16x4,
                        id: ID_MAP[1],
                        count: 1,
                        access: Access::ReadOnly,
                    },
                ],
                output: TensorIr {
                    layout: Layout::from_shape([32]),
                    slice: Slice::from(..),
                    r#type: DataType::F16x4,
                    id: ID_MAP[2],
                    count: 1,
                    access: Access::WriteOnly,
                },
            }),
            Box::new(PhonyBinaryOp {
                id: Default::default(),
                input: [
                    TensorIr {
                        layout: Layout::from_shape([32]),
                        slice: Slice::from(..),
                        r#type: DataType::F32x4,
                        id: ID_MAP[2],
                        count: 2,
                        access: Access::ReadOnly,
                    },
                    TensorIr {
                        layout: Layout::from_shape([32]),
                        slice: Slice::from(..),
                        r#type: DataType::F16x4,
                        id: ID_MAP[3],
                        count: 1,
                        access: Access::ReadOnly,
                    },
                ],
                output: TensorIr {
                    layout: Layout::from_shape([32]),
                    slice: Slice::from(..),
                    r#type: DataType::F32x4,
                    id: ID_MAP[4],
                    count: 1,
                    access: Access::WriteOnly,
                },
            }),
            Box::new(PhonyBinaryOp {
                id: Default::default(),
                input: [
                    TensorIr {
                        layout: Layout::from_shape([32]),
                        slice: Slice::from(..),
                        r#type: DataType::F32x4,
                        id: ID_MAP[4],
                        count: 1,
                        access: Access::ReadOnly,
                    },
                    TensorIr {
                        layout: Layout::from_shape([32]),
                        slice: Slice::from(..),
                        r#type: DataType::F32x4,
                        id: ID_MAP[2],
                        count: 2,
                        access: Access::ReadOnly,
                    },
                ],
                output: TensorIr {
                    layout: Layout::from_shape([32]),
                    slice: Slice::from(..),
                    r#type: DataType::F32x4,
                    id: ID_MAP[5],
                    count: 1,
                    access: Access::WriteOnly,
                },
            }),
            Box::new(PhonyUnaryOp {
                id: Default::default(),
                input: TensorIr {
                    layout: Layout::from_shape([32]),
                    slice: Slice::from(..),
                    r#type: DataType::F32x4,
                    id: ID_MAP[2],
                    count: 1,
                    access: Access::ReadOnly,
                },
                output: TensorIr {
                    layout: Layout::from_shape([32]),
                    slice: Slice::from(..),
                    r#type: DataType::F32x4,
                    id: ID_MAP[6],
                    count: 1,
                    access: Access::WriteOnly,
                },
            }),
        ];
        let id = ID_MAP[5];
        let tape = TensorTape { id, ops };
        let (sender, receiver) = flume::bounded(0);
        cpu.execute(DeviceEvent::Execute { tape, sender });

        let _ = receiver.recv_async().await??;
        Ok(())
    }
}
