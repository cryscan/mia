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
        let layout = io[0].layout.pad_to(2);
        let x = backend.fetch(io[0].id);

        #[cfg(not(feature = "rayon"))]
        let output: Box<_> = handle(move || {
            let (lo, hi) = layout.split_at(0);
            hi.iter_indices()
                .flat_map(|(_, hi)| {
                    let x = x.read_slice::<f32>();
                    let x: Vec<_> = lo.iter_indices().map(move |(_, lo)| x[lo + hi]).collect();
                    let x = x.into_iter();

                    let max = x.clone().fold(f32::NEG_INFINITY, f32::max);
                    let x = x.map(move |x| (x - max).exp());
                    let sum: f32 = x.clone().sum();
                    x.map(move |x| x / sum)
                })
                .collect()
        })
        .await;
        #[cfg(feature = "rayon")]
        let output: Box<_> = handle(move || {
            use rayon::prelude::*;

            let (lo, hi) = layout.split_at(0);
            hi.par_iter_indices()
                .flat_map(|(_, hi)| {
                    let x = x.read_slice::<f32>();
                    let x: Vec<_> = lo.iter_indices().map(move |(_, lo)| x[lo + hi]).collect();
                    let x = x.into_par_iter();

                    let max = x.clone().reduce(|| f32::NEG_INFINITY, f32::max);
                    let x = x.map(move |x| (x - max).exp());
                    let sum: f32 = x.clone().sum();
                    x.map(move |x| x / sum)
                })
                .collect()
        })
        .await;

        let output = output.into_iter().flat_map(|z| z.to_ne_bytes()).collect();
        *backend.fetch(io[1].id).write() = output;
    }
}

impl BackendOp<Backend> for SoftmaxOp<f16> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let layout = io[0].layout.clone();
        let x = backend.fetch(io[0].id);

        #[cfg(not(feature = "rayon"))]
        let output: Box<_> = handle(move || {
            let (lo, hi) = layout.split_at(0);
            hi.iter_indices()
                .flat_map(|(_, hi)| {
                    let x = x.read_slice::<f16>();
                    let x: Vec<_> = lo.iter_indices().map(move |(_, lo)| x[lo + hi]).collect();
                    let x = x.into_iter();

                    let max = x.clone().fold(f16::NEG_INFINITY, f16::max);
                    let x = x.map(move |x| f16::to_f32(x - max).exp());
                    let sum: f32 = x.clone().sum();
                    x.map(move |x| x / sum).map(f16::from_f32)
                })
                .collect()
        })
        .await;
        #[cfg(feature = "rayon")]
        let output: Box<_> = handle(move || {
            use rayon::prelude::*;

            let (lo, hi) = layout.split_at(0);
            hi.par_iter_indices()
                .flat_map(|(_, hi)| {
                    let x = x.read_slice::<f16>();
                    let x: Vec<_> = lo.iter_indices().map(move |(_, lo)| x[lo + hi]).collect();
                    let x = x.into_par_iter();

                    let max = x.clone().reduce(|| f16::NEG_INFINITY, f16::max);
                    let x = x.map(move |x| f16::to_f32(x - max).exp());
                    let sum: f32 = x.clone().sum();
                    x.map(move |x| x / sum).map(f16::from_f32)
                })
                .collect()
        })
        .await;

        let output = output.into_iter().flat_map(|z| z.to_ne_bytes()).collect();
        *backend.fetch(io[1].id).write() = output;
    }
}

impl BackendOp<Backend> for LayerNormOp<f32> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let eps = self.eps;
        let layout = io[0].layout.pad_to(2);
        let x = backend.fetch(io[0].id);
        let w = backend.fetch(io[1].id);
        let b = backend.fetch(io[2].id);

        #[cfg(not(feature = "rayon"))]
        let output: Box<_> = handle(move || {
            let (lo, hi) = layout.split_at(0);
            hi.iter_indices()
                .flat_map(|(_, hi)| {
                    let x = x.read_slice::<f32>();
                    let w = w.read_slice::<f16>();
                    let b = b.read_slice::<f16>();

                    let (mean, m2, count) = lo.iter_indices().map(|(_, lo)| x[lo + hi]).fold(
                        (0.0, 0.0, 0u32),
                        |(mean, m2, count), x| {
                            let count = count + 1;
                            let delta = x - mean;
                            let mean = mean + delta / count as f32;
                            let m2 = m2 + delta * (x - mean);
                            (mean, m2, count)
                        },
                    );
                    let variance = m2 / count as f32 + eps;
                    let deviation = 1.0 / variance.sqrt();

                    let mean = f16::from_f32(mean);
                    let deviation = f16::from_f32(deviation);
                    lo.iter_indices().map(move |(_, lo)| {
                        let x = f16::from_f32(x[lo + hi]);
                        let w = w[lo];
                        let b = b[lo];
                        (x - mean) * deviation * w + b
                    })
                })
                .collect()
        })
        .await;
        #[cfg(feature = "rayon")]
        let output: Box<_> = handle(move || {
            use rayon::prelude::*;

            let (lo, hi) = layout.split_at(0);
            hi.par_iter_indices()
                .flat_map(|(_, hi)| {
                    let (mean, m2, count) = lo
                        .par_iter_indices()
                        .map(|(_, lo)| x.read_slice::<f32>()[lo + hi])
                        .fold(
                            || (0.0, 0.0, 0u32),
                            |(mean, m2, count), x| {
                                let count = count + 1;
                                let delta = x - mean;
                                let mean = mean + delta / count as f32;
                                let m2 = m2 + delta * (x - mean);
                                (mean, m2, count)
                            },
                        )
                        .reduce(
                            || (0.0, 0.0, 0u32),
                            |(mean_1, m2_1, count_1), (mean_2, m2_2, count_2)| {
                                let count = count_1 + count_2;
                                let delta = mean_2 - mean_1;
                                let count_1 = count_1 as f32;
                                let count_2 = count_2 as f32;
                                let mean = match count {
                                    0 => 0.0,
                                    _ => (mean_1 * count_1 + mean_2 * count_2) / count as f32,
                                };
                                let m2 = match count {
                                    0 => 0.0,
                                    _ => m2_1 + m2_2 + delta * delta * (count_1 * count_2),
                                };
                                (mean, m2, count)
                            },
                        );
                    let variance = m2 / count as f32 + eps;
                    let deviation = 1.0 / variance.sqrt();

                    let mean = f16::from_f32(mean);
                    let deviation = f16::from_f32(deviation);

                    let x = x.clone();
                    let w = w.clone();
                    let b = b.clone();
                    lo.par_iter_indices().map(move |(_, lo)| {
                        let x = x.read_slice::<f32>();
                        let w = w.read_slice::<f16>();
                        let b = b.read_slice::<f16>();

                        let x = f16::from_f32(x[lo + hi]);
                        let w = w[lo];
                        let b = b[lo];
                        (x - mean) * deviation * w + b
                    })
                })
                .collect()
        })
        .await;

        let output = output.into_iter().flat_map(|z| z.to_ne_bytes()).collect();
        *backend.fetch(io[3].id).write() = output;
    }
}

impl BackendOp<Backend> for LayerNormOp<f16> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let eps = self.eps;
        let layout = io[0].layout.pad_to(2);
        let x = backend.fetch(io[0].id);
        let w = backend.fetch(io[1].id);
        let b = backend.fetch(io[2].id);

        #[cfg(not(feature = "rayon"))]
        let output: Box<_> = handle(move || {
            let (lo, hi) = layout.split_at(0);
            hi.iter_indices()
                .flat_map(|(_, hi)| {
                    let x = x.read_slice::<f16>();
                    let w = w.read_slice::<f16>();
                    let b = b.read_slice::<f16>();

                    let (mean, m2, count) = lo
                        .iter_indices()
                        .map(|(_, lo)| x[lo + hi])
                        .map(f16::to_f32)
                        .fold((0.0, 0.0, 0u32), |(mean, m2, count), x| {
                            let count = count + 1;
                            let delta = x - mean;
                            let mean = mean + delta / count as f32;
                            let m2 = m2 + delta * (x - mean);
                            (mean, m2, count)
                        });
                    let variance = m2 / count as f32 + eps;
                    let deviation = 1.0 / variance.sqrt();

                    let mean = f16::from_f32(mean);
                    let deviation = f16::from_f32(deviation);
                    lo.iter_indices().map(move |(_, lo)| {
                        let x = x[lo + hi];
                        let w = w[lo];
                        let b = b[lo];
                        (x - mean) * deviation * w + b
                    })
                })
                .collect()
        })
        .await;
        #[cfg(feature = "rayon")]
        let output: Box<_> = handle(move || {
            use rayon::prelude::*;

            let (lo, hi) = layout.split_at(0);
            hi.par_iter_indices()
                .flat_map(|(_, hi)| {
                    let (mean, m2, count) = lo
                        .par_iter_indices()
                        .map(|(_, lo)| x.read_slice::<f16>()[lo + hi])
                        .map(f16::to_f32)
                        .fold(
                            || (0.0, 0.0, 0u32),
                            |(mean, m2, count), x| {
                                let count = count + 1;
                                let delta = x - mean;
                                let mean = mean + delta / count as f32;
                                let m2 = m2 + delta * (x - mean);
                                (mean, m2, count)
                            },
                        )
                        .reduce(
                            || (0.0, 0.0, 0u32),
                            |(mean_1, m2_1, count_1), (mean_2, m2_2, count_2)| {
                                let count = count_1 + count_2;
                                let delta = mean_2 - mean_1;
                                let count_1 = count_1 as f32;
                                let count_2 = count_2 as f32;
                                let mean = match count {
                                    0 => 0.0,
                                    _ => (mean_1 * count_1 + mean_2 * count_2) / count as f32,
                                };
                                let m2 = match count {
                                    0 => 0.0,
                                    _ => m2_1 + m2_2 + delta * delta * (count_1 * count_2),
                                };
                                (mean, m2, count)
                            },
                        );
                    let variance = m2 / count as f32 + eps;
                    let deviation = 1.0 / variance.sqrt();

                    let mean = f16::from_f32(mean);
                    let deviation = f16::from_f32(deviation);

                    let x = x.clone();
                    let w = w.clone();
                    let b = b.clone();
                    lo.par_iter_indices().map(move |(_, lo)| {
                        let x = x.read_slice::<f16>();
                        let w = w.read_slice::<f16>();
                        let b = b.read_slice::<f16>();

                        let x = x[lo + hi];
                        let w = w[lo];
                        let b = b[lo];
                        (x - mean) * deviation * w + b
                    })
                })
                .collect()
        })
        .await;

        let output = output.into_iter().flat_map(|z| z.to_ne_bytes()).collect();
        *backend.fetch(io[3].id).write() = output;
    }
}
