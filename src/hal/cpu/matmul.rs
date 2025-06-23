use half::f16;

use crate::{
    hal::ops::{MatMatFp16Op, MatVecFp16Op},
    loom::{
        device::{
            Backend as _,
            cpu::{Backend, LayoutBuffer},
        },
        num::{F16x4, F32x4},
        ops::{BackendOp, TensorIr},
        platform::handle,
    },
};

impl BackendOp<Backend> for MatMatFp16Op<f16> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let [_a, _b, _c] = self.layouts.clone();
        let [tk, tm, bk, bm] = _a.shape().to_array();
        let [_, tn, _, bn] = _b.shape().to_array();
        assert_eq!([tm, tn, bm, bn], _c.shape().to_array());

        let a = backend.fetch(io[0].id);
        let b = backend.fetch(io[1].id);
        let c = backend.create(io[2].id, LayoutBuffer::<f16>::new(&_c));

        #[cfg(not(feature = "rayon"))]
        handle(move || {
            let a = a.read_layout::<F16x4>(&_a);
            let b = b.read_layout::<F16x4>(&_b);
            let mut c = c.write_layout::<f16>(&_c);

            for (j, i, k) in itertools::iproduct!(0..bn, 0..bm, 0..bk) {
                let mut ta = LayoutBuffer::<F16x4>::new([tk, tm]);
                let mut tb = LayoutBuffer::<F16x4>::new([tk, tn]);

                for (y, x) in itertools::iproduct!(0..tm, 0..tk) {
                    ta[[x, y]] = a[[x, y, k, i]];
                }
                for (y, x) in itertools::iproduct!(0..tn, 0..tk) {
                    tb[[x, y]] = b[[x, y, k, j]];
                }
                for (y, x, z) in itertools::iproduct!(0..tn, 0..tm, 0..tk) {
                    let ra = ta[[z, x]];
                    let rb = tb[[z, y]];
                    c[[x, y, i, j]] += ra.dot(rb);
                }
            }
        })
        .await;
        #[cfg(feature = "rayon")]
        handle(move || {
            use rayon::prelude::*;

            let a = a.read_layout::<F16x4>(&_a);
            let b = b.read_layout::<F16x4>(&_b);

            itertools::iproduct!(0..bn, 0..bm)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(j, i)| {
                    let tc = (0..bk)
                        .into_par_iter()
                        .map(|k| {
                            let mut ta = LayoutBuffer::<F16x4>::new([tk, tm]);
                            let mut tb = LayoutBuffer::<F16x4>::new([tk, tn]);
                            let mut tc = LayoutBuffer::<f16>::new([tm, tn]);

                            for (y, x) in itertools::iproduct!(0..tm, 0..tk) {
                                ta[[x, y]] = a[[x, y, k, i]];
                            }
                            for (y, x) in itertools::iproduct!(0..tn, 0..tk) {
                                tb[[x, y]] = b[[x, y, k, j]];
                            }
                            for (y, x, z) in itertools::iproduct!(0..tn, 0..tm, 0..tk) {
                                let ra = ta[[z, x]];
                                let rb = tb[[z, y]];
                                tc[[x, y]] += ra.dot(rb);
                            }
                            tc.into_inner()
                        })
                        .reduce(
                            || vec![f16::default(); tm * tn].into_boxed_slice(),
                            |a, b| itertools::izip!(a, b).map(|(a, b)| a + b).collect(),
                        );
                    ((i, j), LayoutBuffer::from_data(tc, [tm, tn]))
                })
                .for_each(|((i, j), tc)| {
                    let mut c = c.write_layout::<f16>(&_c);
                    for (y, x) in itertools::iproduct!(0..tn, 0..tm) {
                        c[[x, y, i, j]] = tc[[x, y]];
                    }
                });
        })
        .await;
    }
}

impl BackendOp<Backend> for MatMatFp16Op<f32> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let [_a, _b, _c] = self.layouts.clone();
        let [tk, tm, bk, bm] = _a.shape().to_array();
        let [_, tn, _, bn] = _b.shape().to_array();
        assert_eq!([tm, tn, bm, bn], _c.shape().to_array());

        let a = backend.fetch(io[0].id);
        let b = backend.fetch(io[1].id);
        let c = backend.create(io[2].id, LayoutBuffer::<f32>::new(&_c));

        #[cfg(not(feature = "rayon"))]
        handle(move || {
            let a = a.read_layout::<F16x4>(_a);
            let b = b.read_layout::<F32x4>(_b);
            let mut c = c.write_layout::<f32>(_c);

            for (j, i, k) in itertools::iproduct!(0..bn, 0..bm, 0..bk) {
                let mut ta = LayoutBuffer::<F16x4>::new([tk, tm]);
                let mut tb = LayoutBuffer::<F32x4>::new([tk, tn]);

                for (y, x) in itertools::iproduct!(0..tm, 0..tk) {
                    ta[[x, y]] = a[[x, y, k, i]]
                }
                for (y, x) in itertools::iproduct!(0..tn, 0..tk) {
                    tb[[x, y]] = b[[x, y, k, j]];
                }
                for (y, x, z) in itertools::iproduct!(0..tn, 0..tm, 0..tk) {
                    let ra = ta[[z, x]];
                    let rb = tb[[z, y]];
                    c[[x, y, i, j]] += ra.to_f32().dot(rb);
                }
            }
        })
        .await;
        #[cfg(feature = "rayon")]
        handle(move || {
            use rayon::prelude::*;

            itertools::iproduct!(0..bn, 0..bm)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(move |(j, i)| {
                    let _a = _a.clone();
                    let _b = _b.clone();
                    let a = a.clone();
                    let b = b.clone();

                    let tc = (0..bk)
                        .into_par_iter()
                        .map(move |k| {
                            let mut ta = LayoutBuffer::<F16x4>::new([tk, tm]);
                            let mut tb = LayoutBuffer::<F32x4>::new([tk, tn]);
                            let mut tc = LayoutBuffer::<f32>::new([tm, tn]);

                            let a = a.read_layout::<F16x4>(&_a);
                            let b = b.read_layout::<F32x4>(&_b);

                            for (y, x) in itertools::iproduct!(0..tm, 0..tk) {
                                ta[[x, y]] = a[[x, y, k, i]]
                            }
                            for (y, x) in itertools::iproduct!(0..tn, 0..tk) {
                                tb[[x, y]] = b[[x, y, k, j]];
                            }
                            for (y, x, z) in itertools::iproduct!(0..tn, 0..tm, 0..tk) {
                                let ra = ta[[z, x]];
                                let rb = tb[[z, y]];
                                tc[[x, y]] += ra.to_f32().dot(rb);
                            }
                            tc.into_inner()
                        })
                        .reduce(
                            || vec![f32::default(); tm * tn].into_boxed_slice(),
                            |a, b| itertools::izip!(a, b).map(|(a, b)| a + b).collect(),
                        );
                    ((i, j), LayoutBuffer::from_data(tc, [tm, tn]))
                })
                .for_each(|((i, j), tc)| {
                    let mut c = c.write_layout::<f32>(&_c);
                    for (y, x) in itertools::iproduct!(0..tn, 0..tm) {
                        c[[x, y, i, j]] = tc[[x, y]];
                    }
                });
        })
        .await;
    }
}

impl BackendOp<Backend> for MatVecFp16Op<f16> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let [_a, _b, _c] = self.layouts.clone();
        let [k, m] = _a.shape().to_array();
        let [_, n] = _b.shape().to_array();
        assert_eq!([m, n], _c.shape().to_array());

        let a = backend.fetch(io[0].id);
        let b = backend.fetch(io[1].id);

        #[cfg(not(feature = "rayon"))]
        let output: Vec<_> = handle(move || {
            let a = a.read_layout::<F16x4>(&_a);
            let b = b.read_layout::<F16x4>(&_b);
            itertools::iproduct!(0..n, 0..m)
                .map(|(y, x)| (0..k).fold(f16::default(), |c, z| c + a[[z, x]].dot(b[[z, y]])))
                .collect()
        })
        .await;
        #[cfg(feature = "rayon")]
        let output: Vec<_> = handle(move || {
            use rayon::prelude::*;

            let a = a.read_layout::<F16x4>(&_a);
            let b = b.read_layout::<F16x4>(&_b);
            itertools::iproduct!(0..n, 0..m)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(y, x)| {
                    (0..k)
                        .into_par_iter()
                        .fold(f16::default, |c, z| c + a[[z, x]].dot(b[[z, y]]))
                        .reduce(f16::default, |x, y| x + y)
                })
                .collect()
        })
        .await;

        backend.create(io[2].id, output);
    }
}

impl BackendOp<Backend> for MatVecFp16Op<f32> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let [_a, _b, _c] = self.layouts.clone();
        let [k, m] = _a.shape().to_array();
        let [_, n] = _b.shape().to_array();
        assert_eq!([m, n], _c.shape().to_array());

        let a = backend.fetch(io[0].id);
        let b = backend.fetch(io[1].id);

        #[cfg(not(feature = "rayon"))]
        let output: Vec<_> = handle(move || {
            let a = a.read_layout::<F16x4>(&_a);
            let b = b.read_layout::<F32x4>(&_b);
            itertools::iproduct!(0..n, 0..m)
                .map(|(y, x)| (0..k).fold(0.0, |c, z| c + a[[z, x]].to_f32().dot(b[[z, y]])))
                .collect()
        })
        .await;
        #[cfg(feature = "rayon")]
        let output: Vec<_> = handle(move || {
            use rayon::prelude::*;

            let a = a.read_layout::<F16x4>(&_a);
            let b = b.read_layout::<F32x4>(&_b);
            itertools::iproduct!(0..n, 0..m)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(y, x)| {
                    (0..k)
                        .into_par_iter()
                        .fold(f32::default, |c, z| c + a[[z, x]].to_f32().dot(b[[z, y]]))
                        .reduce(f32::default, |x, y| x + y)
                })
                .collect()
        })
        .await;

        backend.create(io[2].id, output);
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, sync::Arc};

    use half::f16;
    use itertools::Itertools;

    use crate::{
        hal::frontend::MatrixFp16,
        loom::{
            device::{CpuBuilder, Device},
            tensor::Tensor,
        },
    };

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

    async fn test_matmul_f16_f16_inner(
        device: impl Device + Clone,
        k: usize,
        m: usize,
        n: usize,
    ) -> Result<(), Box<dyn Error>> {
        // create test data for matrix A (K x M)
        let a_data: Arc<_> = (0..m)
            .cartesian_product(0..k)
            .map(|(m, k)| fastrand::f32() * 0.1 + m as f32 * 0.01 + k as f32 * 0.001)
            .map(f16::from_f32)
            .collect();
        let a_tensor = Tensor::create(device.clone(), [k, m], a_data.clone())?;
        let a_matrix = MatrixFp16::from_tensor(a_tensor)?;

        // create test data for matrix B (K x N) - note: this is transposed format
        let b_data: Arc<_> = (0..n)
            .cartesian_product(0..k)
            .map(|(n, k)| fastrand::f32() * 0.1 + n as f32 * 0.002 + k as f32 * 0.003)
            .map(f16::from_f32)
            .collect();
        let b_tensor = Tensor::create(device.clone(), [k, n], b_data.clone())?;

        // perform matrix multiplication
        let result = a_matrix.matmul(b_tensor)?;
        let output = result.back().await?;

        // compute reference result using standard matrix multiplication
        // A is K x M, B is K x N (transposed), result should be M x N
        let mut r#ref = vec![f16::ZERO; m * n];
        for (n_idx, m_idx) in itertools::iproduct!(0..n, 0..m) {
            let mut sum = 0.0f32;
            for k_idx in 0..k {
                let a_val = f16::to_f32(a_data[m_idx * k + k_idx]);
                let b_val = f16::to_f32(b_data[n_idx * k + k_idx]);
                sum += a_val * b_val;
            }
            r#ref[n_idx * m + m_idx] = f16::from_f32(sum);
        }

        for (index, (&computed, &expected)) in output.iter().zip_eq(r#ref.iter()).enumerate() {
            assert_approx_eq!(index, f16::to_f32(computed), f16::to_f32(expected), 1e-2);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_matmul_f16_f16() -> Result<(), Box<dyn Error>> {
        fastrand::seed(42);
        let cpu = CpuBuilder::new().add_default_ops().build().await;

        test_matmul_f16_f16_inner(cpu.clone(), 64, 32, 48).await?;
        test_matmul_f16_f16_inner(cpu.clone(), 32, 16, 24).await?;
        test_matmul_f16_f16_inner(cpu.clone(), 64, 32, 1).await?;

        Ok(())
    }

    async fn test_matmul_f16_f32_inner(
        device: impl Device + Clone,
        k: usize,
        m: usize,
        n: usize,
    ) -> Result<(), Box<dyn Error>> {
        // create test data for matrix A (K x M) as f16
        let a_data: Arc<_> = (0..m)
            .cartesian_product(0..k)
            .map(|(m, k)| fastrand::f32() * 0.1 + m as f32 * 0.01 + k as f32 * 0.001)
            .map(f16::from_f32)
            .collect();
        let a_tensor = Tensor::create(device.clone(), [k, m], a_data.clone())?;
        let a_matrix = MatrixFp16::from_tensor(a_tensor)?;

        // create test data for matrix B (K x N) as f32 - note: this is transposed format
        let b_data: Arc<_> = (0..n)
            .cartesian_product(0..k)
            .map(|(n, k)| fastrand::f32() * 0.1 + n as f32 * 0.002 + k as f32 * 0.003)
            .collect();
        let b_tensor = Tensor::create(device.clone(), [k, n], b_data.clone())?;

        // perform matrix multiplication (f16 x f32 -> f32)
        let result = a_matrix.matmul(b_tensor)?;
        let output = result.back().await?;

        // compute reference result using standard matrix multiplication
        // A is K x M (f16), B is K x N (f32), result should be M x N (f32)
        let mut r#ref = vec![0.0f32; m * n];
        for (n_idx, m_idx) in itertools::iproduct!(0..n, 0..m) {
            let mut sum = 0.0f32;
            for k_idx in 0..k {
                let a_val = f16::to_f32(a_data[m_idx * k + k_idx]);
                let b_val = b_data[n_idx * k + k_idx];
                sum += a_val * b_val;
            }
            r#ref[n_idx * m + m_idx] = sum;
        }

        for (index, (&computed, &expected)) in output.iter().zip_eq(r#ref.iter()).enumerate() {
            assert_approx_eq!(index, computed, expected, 1e-4);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_matmul_f16_f32() -> Result<(), Box<dyn Error>> {
        fastrand::seed(42);
        let cpu = CpuBuilder::new().add_default_ops().build().await;

        test_matmul_f16_f32_inner(cpu.clone(), 64, 32, 48).await?;
        test_matmul_f16_f32_inner(cpu.clone(), 32, 16, 24).await?;
        test_matmul_f16_f32_inner(cpu.clone(), 64, 32, 1).await?;

        Ok(())
    }
}
