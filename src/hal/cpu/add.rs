use half::f16;
use itertools::Itertools;

use crate::{
    hal::ops::AddOp,
    loom::{
        device::{Backend as _, cpu::Backend},
        ops::{BackendOp, TensorIr},
    },
};

impl BackendOp<Backend> for AddOp<f32> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let output = {
            let x = backend.fetch(io[0].id);
            let y = backend.fetch(io[1].id);

            let x = x.read_slice::<f32>();
            let y = y.read_slice::<f32>();

            x.iter()
                .zip_eq(y.iter())
                .map(|(x, y)| x + y)
                .flat_map(|z| z.to_ne_bytes())
                .collect()
        };
        *backend.fetch(io[2].id).write() = output;
    }
}

impl BackendOp<Backend> for AddOp<f16> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let output = {
            let x = backend.fetch(io[0].id);
            let y = backend.fetch(io[1].id);

            let x = x.read_slice::<f16>();
            let y = y.read_slice::<f16>();

            x.iter()
                .zip_eq(y.iter())
                .map(|(x, y)| x + y)
                .flat_map(|z| z.to_ne_bytes())
                .collect()
        };
        *backend.fetch(io[2].id).write() = output;
    }
}
