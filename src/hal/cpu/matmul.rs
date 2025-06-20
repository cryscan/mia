use crate::{
    hal::ops::MatMatFp16Op,
    loom::{
        device::{Backend as _, cpu::Backend},
        num::F16x4,
        ops::{BackendOp, TensorIr},
        platform::handle,
    },
};

impl BackendOp<Backend> for MatMatFp16Op<F16x4> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let (_, b) = io[0].layout.pad_to(3).split_at(2);
        let x = backend.fetch(io[0].id);
        let y = backend.fetch(io[1].id);

        let output: Vec<_> = handle(move || {
            b.iter_indices()
                .flat_map(|(_, _batch)| {
                    let _x = x.read_slice::<F16x4>();
                    let _y = y.read_slice::<F16x4>();
                    Vec::<F16x4>::new()
                })
                .collect()
        })
        .await;

        backend.create(io[3].id, output);
    }
}
