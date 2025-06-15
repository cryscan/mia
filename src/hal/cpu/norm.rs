use half::f16;

use crate::{
    hal::ops::LayerNormOp,
    loom::{
        device::{Backend as _, cpu::Backend},
        ops::{BackendOp, TensorIr},
        platform::handle,
    },
};

impl BackendOp<Backend> for LayerNormOp<f32> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let layout = io[0].layout.clone();
        let x = backend.fetch(io[0].id);

        #[cfg(not(feature = "rayon"))]
        let output = handle(move || {
            let x = x.read_slice::<f32>();
            x.chunks_exact(layout.span_of(0))
                .step_by(layout.stride_of(0))
                .flat_map(|x| {
                    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let x = x.iter().map(move |x| (x - max).exp());
                    let sum: f32 = x.clone().sum();
                    x.map(move |x| x / sum)
                })
                .flat_map(|x| x.to_ne_bytes())
                .collect()
        })
        .await;
        #[cfg(feature = "rayon")]
        let output = handle(move || {
            use rayon::prelude::*;

            let x = x.read_slice::<f32>();
            x.par_chunks_exact(layout.span_of(0))
                .step_by(layout.stride_of(0))
                .flat_map(|x| {
                    let max = x.par_iter().copied().reduce(|| f32::NEG_INFINITY, f32::max);
                    let x = x.par_iter().map(move |x| (x - max).exp());
                    let sum: f32 = x.clone().sum();
                    x.map(move |x| x / sum)
                })
                .flat_map(|x| x.to_ne_bytes())
                .collect()
        })
        .await;
        *backend.fetch(io[1].id).write() = output;
    }
}

impl BackendOp<Backend> for LayerNormOp<f16> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let layout = io[0].layout.clone();
        let x = backend.fetch(io[0].id);

        #[cfg(not(feature = "rayon"))]
        let output = handle(move || {
            let x = x.read_slice::<f16>();
            x.chunks_exact(layout.span_of(0))
                .step_by(layout.stride_of(0))
                .flat_map(|x| {
                    let max = x.iter().cloned().fold(f16::NEG_INFINITY, f16::max);
                    let x = x.iter().map(move |x| f16::to_f32(x - max).exp());
                    let sum: f32 = x.clone().sum();
                    x.map(move |x| x / sum)
                })
                .flat_map(|x| f16::from_f32(x).to_ne_bytes())
                .collect()
        })
        .await;
        #[cfg(feature = "rayon")]
        let output = handle(move || {
            use rayon::prelude::*;

            let x = x.read_slice::<f16>();
            x.par_chunks_exact(layout.span_of(0))
                .step_by(layout.stride_of(0))
                .flat_map(|x| {
                    let max = x.par_iter().copied().reduce(|| f16::NEG_INFINITY, f16::max);
                    let x = x.par_iter().map(move |x| f16::to_f32(x - max).exp());
                    let sum: f32 = x.clone().sum();
                    x.map(move |x| x / sum)
                })
                .flat_map(|x| f16::from_f32(x).to_ne_bytes())
                .collect()
        })
        .await;
        *backend.fetch(io[1].id).write() = output;
    }
}
