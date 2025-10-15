use std::{
    any::TypeId,
    borrow::Cow,
    cell::{RefCell, RefMut},
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock},
};

use itertools::Itertools;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use super::{
    BackData, Backend as _, Device, DeviceEvent, DeviceId, ExecuteData, OpVTable,
    allocator::{AllocOp, Allocator, StashId},
};
use crate::loom::{
    layout::{IndexFn, IntoLayout, Layout},
    num::Scalar,
    ops::{BackendOp, TensorIr, TensorOp},
    platform,
    tensor::TensorId,
};

#[derive(Debug, Clone)]
pub struct Buffer {
    data: Arc<RwLock<Box<[u8]>>>,
    align: usize,
}

impl<T: Scalar> From<Box<[T]>> for Buffer {
    fn from(value: Box<[T]>) -> Self {
        let size = size_of_val(&value[..]);
        let ptr = Box::leak(value) as *mut [T] as *mut u8;
        let boxed = unsafe {
            // SAFETY: The pointer must be valid and aligned for `u8` and must not be null.
            let slice = ::core::slice::from_raw_parts(ptr, size);
            Box::from(slice)
        };
        let data = Arc::new(RwLock::new(boxed));
        let align = align_of::<T>();
        Self { data, align }
    }
}

impl<T: Scalar> From<Vec<T>> for Buffer {
    fn from(value: Vec<T>) -> Self {
        value.into_boxed_slice().into()
    }
}

impl Buffer {
    #[inline]
    pub fn align(&self) -> usize {
        self.align
    }

    #[inline]
    pub fn read(&self) -> impl Deref<Target = Box<[u8]>> {
        self.data.read().expect("failed to lock buffer")
    }

    #[inline]
    pub fn write(&self) -> impl DerefMut<Target = Box<[u8]>> {
        self.data.write().expect("failed to lock buffer")
    }

    #[inline]
    pub fn read_slice<T: Scalar>(&self) -> impl Deref<Target = [T]> {
        let lock = self.read();
        let phantom = PhantomData;
        BufferReader { lock, phantom }
    }

    #[inline]
    pub fn write_slice<T: Scalar>(&self) -> impl DerefMut<Target = [T]> {
        let lock = self.write();
        let phantom = PhantomData;
        BufferWriter { lock, phantom }
    }

    #[inline]
    pub fn read_layout<T: Scalar>(
        &self,
        layout: impl IntoLayout,
    ) -> LayoutBuffer<impl Deref<Target = [T]>, T> {
        let inner = self.read_slice::<T>();
        let layout = layout.into_layout();
        let phantom = PhantomData::<T>;
        LayoutBuffer {
            inner,
            layout,
            phantom,
        }
    }

    #[inline]
    pub fn write_layout<T: Scalar>(
        &self,
        layout: impl IntoLayout,
    ) -> LayoutBuffer<impl DerefMut<Target = [T]>, T> {
        let inner = self.write_slice::<T>();
        let layout = layout.into_layout();
        let phantom = PhantomData::<T>;
        LayoutBuffer {
            inner,
            layout,
            phantom,
        }
    }

    #[inline]
    pub fn into_inner(self) -> Box<[u8]> {
        match Arc::try_unwrap(self.data) {
            Ok(inner) => inner.into_inner().expect("lock poisoned"),
            Err(data) => unsafe {
                use std::alloc::{Layout, alloc_zeroed};

                // SAFETY: We are guaranteed that the data is valid and properly aligned because it was
                // created from a Box<[T]> and the alignment is preserved.
                let data = data.read().expect("failed to lock buffer");
                let size = data.len().max(64);
                let align = self.align;
                let layout = Layout::from_size_align(size, align).expect("invalid layout");
                let ptr = alloc_zeroed(layout);
                ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
                Box::from_raw(std::slice::from_raw_parts_mut(ptr, size))
            },
        }
    }
}

pub struct BufferReader<R, T> {
    lock: R,
    phantom: PhantomData<T>,
}

impl<R, T> std::ops::Deref for BufferReader<R, T>
where
    R: Deref<Target = Box<[u8]>>,
    T: Scalar,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        bytemuck::cast_slice(&self.lock)
    }
}

pub struct BufferWriter<W, T> {
    lock: W,
    phantom: PhantomData<T>,
}

impl<R, T> std::ops::Deref for BufferWriter<R, T>
where
    R: Deref<Target = Box<[u8]>>,
    T: Scalar,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        bytemuck::cast_slice(&self.lock)
    }
}

impl<R, T> std::ops::DerefMut for BufferWriter<R, T>
where
    R: DerefMut<Target = Box<[u8]>>,
    T: Scalar,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        bytemuck::cast_slice_mut(&mut self.lock)
    }
}

pub struct LayoutBuffer<R, T> {
    inner: R,
    layout: Layout,
    phantom: PhantomData<T>,
}

impl<R, T> LayoutBuffer<R, T> {
    #[inline]
    pub fn into_inner(self) -> R {
        self.inner
    }

    #[inline]
    pub fn layout(&self) -> Layout {
        self.layout.clone()
    }
}

impl<R, T> std::ops::Deref for LayoutBuffer<R, T>
where
    R: Deref<Target = [T]>,
    T: Scalar,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl<R, T> std::ops::DerefMut for LayoutBuffer<R, T>
where
    R: DerefMut<Target = [T]>,
    T: Scalar,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

impl<const N: usize, R, T> std::ops::Index<[usize; N]> for LayoutBuffer<R, T>
where
    R: Deref<Target = [T]>,
    T: Scalar,
{
    type Output = T;

    fn index(&self, index: [usize; N]) -> &Self::Output {
        &self.inner[self.layout.value(index)]
    }
}

impl<const N: usize, R, T> std::ops::IndexMut<[usize; N]> for LayoutBuffer<R, T>
where
    R: DerefMut<Target = [T]>,
    T: Scalar,
{
    fn index_mut(&mut self, index: [usize; N]) -> &mut Self::Output {
        &mut self.inner[self.layout.value(index)]
    }
}

impl<T: Scalar> LayoutBuffer<Box<[T]>, T> {
    #[inline]
    pub fn new(layout: impl IntoLayout) -> Self {
        let layout = layout.into_layout();
        let inner = vec![T::zero(); layout.co_size()].into_boxed_slice();
        let phantom = PhantomData;
        Self {
            inner,
            layout,
            phantom,
        }
    }

    #[inline]
    pub fn from_data(data: Box<[T]>, layout: impl IntoLayout) -> Self {
        let layout = layout.into_layout();
        let inner = data;
        let phantom = PhantomData;
        Self {
            inner,
            layout,
            phantom,
        }
    }
}

impl<T: Scalar> From<LayoutBuffer<Box<[T]>, T>> for Cow<'_, [T]> {
    fn from(value: LayoutBuffer<Box<[T]>, T>) -> Self {
        Cow::Owned(value.inner.into_vec())
    }
}

#[allow(unused)]
pub struct Backend {
    /// Operators that the device is able to execute.
    ops: OpVTable<Self>,
    /// Allocator that tracks buffer renaming.
    allocator: RefCell<Allocator>,
    /// Stack of CPU buffers.
    buffers: HashMap<StashId, Buffer>,
}

impl super::Backend for Backend {
    type Data = Buffer;

    #[inline]
    async fn execute(&mut self, op: &dyn TensorOp, io: Vec<TensorIr>) {
        let id = &op.type_id();
        match self.ops.get(id) {
            Some(f) => f(self, op, io).await,
            #[cfg(not(feature = "strict"))]
            None => log::error!("unable to execute op of type {}", op.name()),
            #[cfg(feature = "strict")]
            None => panic!("unable to execute op of type {}", op.name()),
        }
    }

    #[inline]
    fn create<'a, T, C>(&mut self, id: TensorId, contents: C) -> Self::Data
    where
        T: Scalar,
        C: Into<Cow<'a, [T]>>,
    {
        let id = self.allocator().retrieve(id);
        let data = Self::Data::from(contents.into().into_owned());
        self.buffers.insert(id, data.clone());
        data
    }

    #[inline]
    fn alloc<T: Scalar>(&mut self, id: TensorId, len: usize) -> Self::Data {
        let id = self.allocator().retrieve(id);
        let data = self
            .buffers
            .get(&id)
            .cloned()
            .filter(|data| data.align() == align_of::<T>())
            .filter(|data| data.read_slice::<T>().len() == len);
        let data = match data {
            Some(data) => data,
            None => vec![T::zero(); len].into(),
        };
        self.buffers.insert(id, data.clone());
        data
    }

    #[inline]
    fn try_fetch(&self, id: TensorId) -> Option<Self::Data> {
        let id = self.allocator().retrieve(id);
        self.buffers.get(&id).cloned()
    }
}

impl Backend {
    #[inline]
    pub fn allocator(&self) -> RefMut<'_, Allocator> {
        self.allocator.borrow_mut()
    }
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct Cpu {
    /// The unique identifier of the device.
    id: DeviceId,
    /// Sends ops to execute to the backend.
    sender: flume::Sender<DeviceEvent>,
}

impl Device for Cpu {
    fn execute(&self, event: DeviceEvent) {
        _ = self.sender.send(event)
    }
}

#[derive(Default)]
pub struct CpuBuilder {
    pub ops: OpVTable<Backend>,
}

impl CpuBuilder {
    pub fn new() -> Self {
        Self::default().add_op::<AllocOp>()
    }

    pub async fn build(self) -> Cpu {
        let ops = self.ops;
        let allocator = RefCell::new(Allocator::default());
        let buffers = HashMap::default();

        let (sender, receiver) = flume::unbounded();
        let backend = Backend {
            ops,
            allocator,
            buffers,
        };
        platform::spawn(serve(backend, receiver));

        let id = Default::default();
        Cpu { id, sender }
    }

    pub fn add_op<Op>(mut self) -> Self
    where
        Op: TensorOp + BackendOp<Backend>,
    {
        self.ops.insert(TypeId::of::<Op>(), |backend, op, io| {
            match op.downcast_ref::<Op>() {
                Some(op) => Box::pin(op.execute(backend, io)),
                None => unreachable!(),
            }
        });
        self
    }
}

async fn serve(mut backend: Backend, receiver: flume::Receiver<DeviceEvent>) {
    let mut commit = HashSet::default();

    while let Ok(event) = receiver.recv_async().await {
        match event {
            DeviceEvent::Execute { tape, sender } => {
                let data = async {
                    let ops = tape
                        .ops
                        .iter()
                        .filter(|op| !commit.contains(&op.id()))
                        .cloned()
                        .collect_vec();
                    commit.extend(ops.iter().map(|op| op.id()));

                    for op in ops {
                        let op = backend.allocator().alloc(op)?;
                        backend.execute(&op, op.io()).await;
                    }

                    let data = backend.allocator().mermaid(&tape);
                    Ok(ExecuteData(data))
                }
                .await;
                _ = sender.send_async(data).await
            }
            DeviceEvent::Back { tape, sender } => {
                let data = async {
                    let ops = tape
                        .ops
                        .iter()
                        .filter(|op| !commit.contains(&op.id()))
                        .cloned()
                        .collect_vec();
                    commit.extend(ops.iter().map(|op| op.id()));

                    for op in ops {
                        let op = backend.allocator().alloc(op)?;
                        backend.execute(&op, op.io()).await;
                    }

                    Ok(backend.fetch(tape.id))
                }
                .await
                .map(Buffer::into_inner)
                .map(Arc::from)
                .map(BackData);
                _ = sender.send_async(data).await
            }
            DeviceEvent::Cleanup { retain } => {
                // remove all buffers in the stack unless retained
                let f = |ir: TensorIr| backend.allocator().retrieve(ir.id);
                let f = |op: &dyn TensorOp| op.io().into_iter().map(f);
                let ids: HashSet<_> = retain
                    .iter()
                    .flat_map(|tape| tape.ops.iter().map(AsRef::as_ref).flat_map(f))
                    .collect();
                backend.buffers.retain(|id, _| ids.contains(id));

                // removes all tensors tracked in the allocator unless related to retained ones
                let f = |ir: TensorIr| ir.id;
                let f = |op: &dyn TensorOp| op.io().into_iter().map(f);
                let ids = retain
                    .iter()
                    .flat_map(|tape| tape.ops.iter().map(AsRef::as_ref).flat_map(f))
                    .collect_vec();
                backend.allocator().retain(ids);

                // remove all committed ops unless retained
                let ids: HashSet<_> = retain
                    .iter()
                    .flat_map(|tape| tape.ops.iter().map(|op| op.id()))
                    .collect();
                commit.retain(|id| ids.contains(id));
            }
        }
    }
}
