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
        let output: Vec<_> = handle(move || {
            let (lo, hi) = layout.split_at(1);
            hi.iter_indices()
                .flat_map(|(_, hi)| {
                    let x = x.read_slice::<f32>();
                    let max = lo
                        .iter_indices()
                        .map(|(_, lo)| x[lo + hi])
                        .fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = lo
                        .iter_indices()
                        .map(|(_, lo)| x[lo + hi] - max)
                        .map(f32::exp)
                        .sum();
                    lo.iter_indices()
                        .map(move |(_, lo)| x[lo + hi] - max)
                        .map(move |x| x / exp_sum)
                })
                .collect()
        })
        .await;
        #[cfg(feature = "rayon")]
        let output: Vec<_> = handle(move || {
            use rayon::prelude::*;

            let (lo, hi) = layout.split_at(1);
            hi.par_iter_indices()
                .flat_map(|(_, hi)| {
                    let max = lo
                        .par_iter_indices()
                        .map(|(_, lo)| x.read_slice::<f32>()[lo + hi])
                        .reduce(|| f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = lo
                        .par_iter_indices()
                        .map(|(_, lo)| x.read_slice::<f32>()[lo + hi] - max)
                        .map(f32::exp)
                        .sum();

                    let x = x.clone();
                    lo.par_iter_indices()
                        .map(move |(_, lo)| x.read_slice::<f32>()[lo + hi] - max)
                        .map(move |x| x / exp_sum)
                })
                .collect()
        })
        .await;

        backend.create(io[1].id, output);
    }
}

impl BackendOp<Backend> for SoftmaxOp<f16> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let layout = io[0].layout.clone();
        let x = backend.fetch(io[0].id);

        #[cfg(not(feature = "rayon"))]
        let output: Vec<_> = handle(move || {
            let (lo, hi) = layout.split_at(1);
            hi.iter_indices()
                .flat_map(|(_, hi)| {
                    let x = x.read_slice::<f16>();
                    let max = lo
                        .iter_indices()
                        .map(|(_, lo)| x[lo + hi])
                        .fold(f16::NEG_INFINITY, f16::max);
                    let exp_sum: f32 = lo
                        .iter_indices()
                        .map(|(_, lo)| x[lo + hi] - max)
                        .map(f16::to_f32)
                        .map(f32::exp)
                        .sum();
                    lo.iter_indices()
                        .map(move |(_, lo)| x[lo + hi] - max)
                        .map(f16::to_f32)
                        .map(move |x| x / exp_sum)
                        .map(f16::from_f32)
                })
                .collect()
        })
        .await;
        #[cfg(feature = "rayon")]
        let output: Vec<_> = handle(move || {
            use rayon::prelude::*;

            let (lo, hi) = layout.split_at(1);
            hi.par_iter_indices()
                .flat_map(|(_, hi)| {
                    let max = lo
                        .par_iter_indices()
                        .map(|(_, lo)| x.read_slice::<f16>()[lo + hi])
                        .reduce(|| f16::NEG_INFINITY, f16::max);
                    let exp_sum: f32 = lo
                        .par_iter_indices()
                        .map(|(_, lo)| x.read_slice::<f16>()[lo + hi] - max)
                        .map(f16::to_f32)
                        .map(f32::exp)
                        .sum();

                    let x = x.clone();
                    lo.par_iter_indices()
                        .map(move |(_, lo)| x.read_slice::<f16>()[lo + hi] - max)
                        .map(f16::to_f32)
                        .map(move |x| x / exp_sum)
                        .map(f16::from_f32)
                })
                .collect()
        })
        .await;

        backend.create(io[1].id, output);
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
        let output: Vec<_> = handle(move || {
            let (lo, hi) = layout.split_at(1);
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
        let output: Vec<_> = handle(move || {
            use rayon::prelude::*;

            let (lo, hi) = layout.split_at(1);
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

        backend.create(io[3].id, output);
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
        let output: Vec<_> = handle(move || {
            let (lo, hi) = layout.split_at(1);
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
        let output: Vec<_> = handle(move || {
            use rayon::prelude::*;

            let (lo, hi) = layout.split_at(1);
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

        backend.create(io[3].id, output);
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, sync::Arc};

    use half::f16;

    use crate::loom::{device::CpuBuilder, tensor::Tensor};

    macro_rules! assert_approx_eq {
        ($i:expr, $a:expr, $b:expr, $eps:expr) => {
            assert!(
                ($a - $b).abs() < $eps,
                "assertion failed at {}: `(left ~= right)`\n  left: `{}`\n right: `{}`",
                $i,
                $a,
                $b
            );
        };
    }

    #[tokio::test]
    async fn test_softmax_f16() -> Result<(), Box<dyn Error>> {
        fastrand::seed(42);

        let cpu = CpuBuilder::new().add_default_ops().build().await;
        const C: usize = 1024;
        const T: usize = 768;

        let data: Arc<_> = (0..C * T)
            .map(|_| fastrand::f32())
            .map(f16::from_f32)
            .collect();
        let a = Tensor::create(cpu.clone(), [C, T], data.clone());
        let a = a.softmax();

        let output = a.back().await?;
        let r#ref: Box<_> = data
            .chunks_exact(T)
            .flat_map(|x| {
                let max = x.iter().copied().fold(f16::NEG_INFINITY, f16::max);
                let exp_sum: f32 = x
                    .iter()
                    .map(|v| v - max)
                    .map(f16::to_f32)
                    .map(f32::exp)
                    .sum();
                x.iter()
                    .map(|v| v - max)
                    .map(f16::to_f32)
                    .map(f32::exp)
                    .map(move |v| v / exp_sum)
                    .map(f16::from_f32)
                    .collect::<Box<_>>()
            })
            .collect();

        for (index, (&a, &b)) in output.iter().zip(r#ref.iter()).enumerate() {
            assert_approx_eq!(index, f16::to_f32(a), f16::to_f32(b), 1e-2);
        }

        let data: Arc<_> = (0..C).map(|_| fastrand::f32()).map(f16::from_f32).collect();
        let a = Tensor::create(cpu.clone(), [C], data.clone());
        let a = a.softmax();

        let output = a.back().await?;
        let r#ref: Box<_> = {
            let max = data.iter().copied().fold(f16::NEG_INFINITY, f16::max);
            let exp_sum: f32 = data
                .iter()
                .map(|v| v - max)
                .map(f16::to_f32)
                .map(f32::exp)
                .sum();
            data.iter()
                .map(|v| v - max)
                .map(f16::to_f32)
                .map(f32::exp)
                .map(move |v| v / exp_sum)
                .map(f16::from_f32)
                .collect()
        };

        for (index, (&a, &b)) in output.iter().zip(r#ref.iter()).enumerate() {
            assert_approx_eq!(index, f16::to_f32(a), f16::to_f32(b), 1e-2);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_softmax_f32() -> Result<(), Box<dyn Error>> {
        fastrand::seed(42);

        let cpu = CpuBuilder::new().add_default_ops().build().await;
        const C: usize = 1024;
        const T: usize = 768;

        let data: Arc<_> = (0..C * T).map(|_| fastrand::f32()).collect();
        let a = Tensor::create(cpu.clone(), [C, T], data.clone());
        let a = a.softmax();

        let output = a.back().await?;
        let r#ref: Box<_> = data
            .chunks_exact(T)
            .flat_map(|x| {
                let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = x.iter().map(|v| v - max).map(f32::exp).sum();
                x.iter()
                    .map(|v| v - max)
                    .map(f32::exp)
                    .map(move |v| v / exp_sum)
                    .collect::<Box<_>>()
            })
            .collect();

        for (index, (&a, &b)) in output.iter().zip(r#ref.iter()).enumerate() {
            assert_approx_eq!(index, a, b, 1e-2);
        }

        let data: Arc<_> = (0..C).map(|_| fastrand::f32()).collect();
        let a = Tensor::create(cpu.clone(), [C], data.clone());
        let a = a.softmax();

        let output = a.back().await?;
        let r#ref: Box<_> = {
            let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = data.iter().map(|v| v - max).map(f32::exp).sum();
            data.iter()
                .map(|v| v - max)
                .map(f32::exp)
                .map(move |v| v / exp_sum)
                .collect()
        };

        for (index, (&a, &b)) in output.iter().zip(r#ref.iter()).enumerate() {
            assert_approx_eq!(index, a, b, 1e-2);
        }

        Ok(())
    }
}
