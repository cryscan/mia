use half::f16;

use crate::{
    hal::ops::{LayerNormOp, SoftmaxOp},
    loom::{
        device::{Backend as _, cpu::Backend},
        ops::{BackendOp, TensorIr},
        platform::handle,
    },
};

impl BackendOp<Backend> for SoftmaxOp<f32> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let layout = io[0].layout.clone();
        let x = backend.fetch(io[0].id);

        #[cfg(not(feature = "rayon"))]
        let output = handle(move || {
            let x = x.read_slice::<f32>();
            x.chunks_exact(layout.span_of(0))
                .map(|x| x.iter().step_by(layout.stride_of(0)))
                .flat_map(|x| {
                    let max = x.clone().copied().fold(f32::NEG_INFINITY, f32::max);
                    let x = x.map(move |x| (x - max).exp());
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
                .map(|x| x.par_iter().step_by(layout.stride_of(0)))
                .flat_map(|x| {
                    let max = x.clone().copied().reduce(|| f32::NEG_INFINITY, f32::max);
                    let x = x.map(move |x| (x - max).exp());
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

impl BackendOp<Backend> for SoftmaxOp<f16> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let layout = io[0].layout.clone();
        let x = backend.fetch(io[0].id);

        #[cfg(not(feature = "rayon"))]
        let output = handle(move || {
            let x = x.read_slice::<f16>();
            x.chunks_exact(layout.span_of(0))
                .map(|x| x.iter().step_by(layout.stride_of(0)))
                .flat_map(|x| {
                    let max = x.clone().copied().fold(f16::NEG_INFINITY, f16::max);
                    let x = x.map(move |x| f16::to_f32(x - max).exp());
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
                .map(|x| x.par_iter().step_by(layout.stride_of(0)))
                .flat_map(|x| {
                    let max = x.clone().copied().reduce(|| f16::NEG_INFINITY, f16::max);
                    let x = x.map(move |x| f16::to_f32(x - max).exp());
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

impl BackendOp<Backend> for LayerNormOp<f32> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let eps = self.eps;
        let layout = io[0].layout.clone();
        let x = backend.fetch(io[0].id);
        let w = backend.fetch(io[1].id);
        let b = backend.fetch(io[2].id);

        let output = handle(move || {
            let x = x.read_slice::<f32>();
            let w = w.read_slice::<f16>();
            let b = b.read_slice::<f16>();

            let x = x.chunks_exact(layout.span_of(0));

            let x = x.map(|x| x.iter().step_by(layout.stride_of(0)));
            let w = w.iter().step_by(layout.stride_of(0));
            let b = b.iter().step_by(layout.stride_of(0));

            itertools::izip!(x, w, b)
                .skip(layout.stride_of(0))
                .flat_map(|(x, &w, &b)| {
                    let (mean, m2, count) =
                        x.clone().fold((0.0, 0.0, 0u32), |(mean, m2, count), x| {
                            let count = count + 1;
                            let delta = x - mean;
                            let mean = mean + delta / count as f32;
                            let m2 = m2 + delta * (x - mean);
                            (mean, m2, count)
                        });
                    let variance = m2 / count as f32 + eps;
                    let deviation = 1.0 / variance.sqrt();
                    let w = f16::to_f32(w);
                    let b = f16::to_f32(b);
                    x.map(move |&x| (x - mean) * deviation * w + b)
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
        let eps = self.eps;
        let layout = io[0].layout.clone();
        let x = backend.fetch(io[0].id);
        let w = backend.fetch(io[1].id);
        let b = backend.fetch(io[2].id);

        let output = handle(move || {
            let x = x.read_slice::<f16>();
            let w = w.read_slice::<f16>();
            let b = b.read_slice::<f16>();

            let x = x.chunks_exact(layout.span_of(0));

            let x = x.map(|x| x.iter().step_by(layout.stride_of(0)));
            let w = w.iter().step_by(layout.stride_of(0));
            let b = b.iter().step_by(layout.stride_of(0));

            itertools::izip!(x, w, b)
                .skip(layout.stride_of(0))
                .flat_map(|(x, w, b)| {
                    let (mean, m2, count) =
                        x.clone().fold((0.0, 0.0, 0u32), |(mean, m2, count), &x| {
                            let x = f16::to_f32(x);
                            let count = count + 1;
                            let delta = x - mean;
                            let mean = mean + delta / count as f32;
                            let m2 = m2 + delta * (x - mean);
                            (mean, m2, count)
                        });
                    let variance = m2 / count as f32 + eps;
                    let deviation = f16::from_f32(1.0 / variance.sqrt());
                    let mean = f16::from_f32(mean);
                    x.map(move |&x| (x - mean) * deviation * w + b)
                })
                .flat_map(|x| x.to_ne_bytes())
                .collect()
        })
        .await;
        *backend.fetch(io[1].id).write() = output;
    }
}
