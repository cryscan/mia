use half::f16;

use crate::{
    hal::ops::MatMatFp16Op,
    loom::{
        device::{Backend as _, cpu::Backend},
        layout::{IndexFn, Layout},
        num::{F16x4, F32x4},
        ops::{BackendOp, TensorIr},
        platform::handle,
    },
};

impl BackendOp<Backend> for MatMatFp16Op<f16> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let a = backend.fetch(io[0].id);
        let b = backend.fetch(io[1].id);

        let [_a, _b, _c] = self.layouts.clone();
        let [tk, tm, bk, bm] = _a.shape().to_array();
        let [_, tn, _, bn] = _b.shape().to_array();
        assert_eq!([tm, tn, bm, bn], _c.shape().to_array::<4>());

        #[cfg(not(feature = "rayon"))]
        let output: Vec<_> = handle(move || {
            let a = a.read_slice::<F16x4>();
            let b = b.read_slice::<F16x4>();
            let mut c: Vec<f16> = _c.make_vec();

            for (j, i, k) in itertools::iproduct!(0..bn, 0..bm, 0..bk) {
                let _ta = Layout::from_shape([tk, tm]);
                let _tb = Layout::from_shape([tk, tn]);
                let mut ta = _ta.make_vec();
                let mut tb = _tb.make_vec();

                for (y, x) in itertools::iproduct!(0..tm, 0..tk) {
                    ta[_ta.value([x, y])] = a[_a.value([x, y, k, i])]
                }
                for (y, x) in itertools::iproduct!(0..tn, 0..tk) {
                    tb[_tb.value([x, y])] = b[_b.value([x, y, k, j])];
                }
                for (y, x, z) in itertools::iproduct!(0..tn, 0..tm, 0..tk) {
                    let ra = ta[_ta.value([z, x])];
                    let rb = tb[_tb.value([z, y])];
                    c[_c.value([x, y, i, j])] += ra.dot(rb);
                }
            }
            c
        })
        .await;
        #[cfg(feature = "rayon")]
        let output: Vec<_> = handle(move || {
            use itertools::Itertools;
            use rayon::prelude::*;

            let layout = Layout::from_shape([tm, tn, bm, bn]);
            let tiles: Vec<_> = itertools::iproduct!(0..bn, 0..bm)
                .collect_vec()
                .into_par_iter()
                .flat_map(move |(j, i)| {
                    let _a = _a.clone();
                    let _b = _b.clone();
                    let a = a.clone();
                    let b = b.clone();

                    (0..bk)
                        .into_par_iter()
                        .map(move |k| {
                            let _ta = Layout::from_shape([tk, tm]);
                            let _tb = Layout::from_shape([tk, tn]);
                            let _tc = Layout::from_shape([tm, tn]);
                            let mut ta = _ta.make_vec();
                            let mut tb = _tb.make_vec();
                            let mut tc = _tc.make_vec();

                            let a = a.read_slice::<F16x4>();
                            let b = b.read_slice::<F16x4>();

                            for (y, x) in itertools::iproduct!(0..tm, 0..tk) {
                                ta[_ta.value([x, y])] = a[_a.value([x, y, k, i])]
                            }
                            for (y, x) in itertools::iproduct!(0..tn, 0..tk) {
                                tb[_tb.value([x, y])] = b[_b.value([x, y, k, j])];
                            }
                            for (y, x, z) in itertools::iproduct!(0..tn, 0..tm, 0..tk) {
                                let ra = ta[_ta.value([z, x])];
                                let rb = tb[_tb.value([z, y])];
                                tc[_tc.value([x, y])] += ra.dot(rb);
                            }
                            tc
                        })
                        .reduce(
                            || vec![f16::default(); tm * tn],
                            |a, b| a.into_iter().zip_eq(b).map(|(a, b)| a + b).collect(),
                        )
                })
                .collect();

            let mut c = _c.make_vec();
            for (j, i, y, x) in itertools::iproduct!(0..bn, 0..bm, 0..tn, 0..tm) {
                c[_c.value([x, y, i, j])] = tiles[layout.value([x, y, i, j])]
            }
            c
        })
        .await;

        backend.create(io[2].id, output);
    }
}

impl BackendOp<Backend> for MatMatFp16Op<f32> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let a = backend.fetch(io[0].id);
        let b = backend.fetch(io[1].id);

        let [_a, _b, _c] = self.layouts.clone();
        let [tk, tm, bk, bm] = _a.shape().to_array();
        let [_, tn, _, bn] = _b.shape().to_array();
        assert_eq!([tm, tn, bm, bn], _c.shape().to_array::<4>());

        #[cfg(not(feature = "rayon"))]
        let output: Vec<_> = handle(move || {
            let a = a.read_slice::<F16x4>();
            let b = b.read_slice::<F32x4>();
            let mut c: Vec<f32> = _c.make_vec();

            for (j, i, k) in itertools::iproduct!(0..bn, 0..bm, 0..bk) {
                let _ta = Layout::from_shape([tk, tm]);
                let _tb = Layout::from_shape([tk, tn]);
                let mut ta = _ta.make_vec();
                let mut tb = _tb.make_vec();

                for (y, x) in itertools::iproduct!(0..tm, 0..tk) {
                    ta[_ta.value([x, y])] = a[_a.value([x, y, k, i])]
                }
                for (y, x) in itertools::iproduct!(0..tn, 0..tk) {
                    tb[_tb.value([x, y])] = b[_b.value([x, y, k, j])];
                }
                for (y, x, z) in itertools::iproduct!(0..tn, 0..tm, 0..tk) {
                    let ra = ta[_ta.value([z, x])];
                    let rb = tb[_tb.value([z, y])];
                    c[_c.value([x, y, i, j])] += ra.to_f32().dot(rb);
                }
            }
            c
        })
        .await;
        #[cfg(feature = "rayon")]
        let output: Vec<_> = handle(move || {
            use itertools::Itertools;
            use rayon::prelude::*;

            let layout = Layout::from_shape([tm, tn, bm, bn]);
            let tiles: Vec<_> = itertools::iproduct!(0..bn, 0..bm)
                .collect_vec()
                .into_par_iter()
                .flat_map(move |(j, i)| {
                    let _a = _a.clone();
                    let _b = _b.clone();
                    let a = a.clone();
                    let b = b.clone();

                    (0..bk)
                        .into_par_iter()
                        .map(move |k| {
                            let _ta = Layout::from_shape([tk, tm]);
                            let _tb = Layout::from_shape([tk, tn]);
                            let _tc = Layout::from_shape([tm, tn]);
                            let mut ta = _ta.make_vec();
                            let mut tb = _tb.make_vec();
                            let mut tc = _tc.make_vec();

                            let a = a.read_slice::<F16x4>();
                            let b = b.read_slice::<F32x4>();

                            for (y, x) in itertools::iproduct!(0..tm, 0..tk) {
                                ta[_ta.value([x, y])] = a[_a.value([x, y, k, i])]
                            }
                            for (y, x) in itertools::iproduct!(0..tn, 0..tk) {
                                tb[_tb.value([x, y])] = b[_b.value([x, y, k, j])];
                            }
                            for (y, x, z) in itertools::iproduct!(0..tn, 0..tm, 0..tk) {
                                let ra = ta[_ta.value([z, x])];
                                let rb = tb[_tb.value([z, y])];
                                tc[_tc.value([x, y])] += ra.to_f32().dot(rb);
                            }
                            tc
                        })
                        .reduce(
                            || vec![f32::default(); tm * tn],
                            |a, b| a.into_iter().zip_eq(b).map(|(a, b)| a + b).collect(),
                        )
                })
                .collect();

            let mut c = _c.make_vec();
            for (j, i, y, x) in itertools::iproduct!(0..bn, 0..bm, 0..tn, 0..tm) {
                c[_c.value([x, y, i, j])] = tiles[layout.value([x, y, i, j])]
            }
            c
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
        loom::{device::CpuBuilder, tensor::Tensor},
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

    #[tokio::test]
    async fn test_matmul_f16_f16() -> Result<(), Box<dyn Error>> {
        fastrand::seed(42);

        let cpu = CpuBuilder::new().add_default_ops().build().await;
        const K: usize = 64;
        const M: usize = 32;
        const N: usize = 48;

        // create test data for matrix A (K x M)
        let a_data: Arc<_> = (0..M)
            .cartesian_product(0..K)
            .map(|(m, k)| fastrand::f32() * 0.1 + m as f32 * 0.01 + k as f32 * 0.001)
            .map(f16::from_f32)
            .collect();
        let a_tensor = Tensor::create(cpu.clone(), [K, M], a_data.clone())?;
        let a_matrix = MatrixFp16::from_tensor(a_tensor)?;

        // create test data for matrix B (K x N) - note: this is transposed format
        let b_data: Arc<_> = (0..N)
            .cartesian_product(0..K)
            .map(|(n, k)| fastrand::f32() * 0.1 + n as f32 * 0.002 + k as f32 * 0.003)
            .map(f16::from_f32)
            .collect();
        let b_tensor = Tensor::create(cpu.clone(), [K, N], b_data.clone())?;

        // perform matrix multiplication
        let result = a_matrix.matmul(b_tensor)?;
        let output = result.back().await?;

        // compute reference result using standard matrix multiplication
        // A is K x M, B is K x N (transposed), result should be M x N
        let mut r#ref = vec![f16::ZERO; M * N];
        for (n, m) in itertools::iproduct!(0..N, 0..M) {
            let mut sum = 0.0f32;
            for k in 0..K {
                let a_val = f16::to_f32(a_data[m * K + k]);
                let b_val = f16::to_f32(b_data[n * K + k]);
                sum += a_val * b_val;
            }
            r#ref[n * M + m] = f16::from_f32(sum);
        }

        for (index, (&computed, &expected)) in output.iter().zip_eq(r#ref.iter()).enumerate() {
            assert_approx_eq!(index, f16::to_f32(computed), f16::to_f32(expected), 1e-2);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_matmul_f16_f32() -> Result<(), Box<dyn Error>> {
        fastrand::seed(42);

        let cpu = CpuBuilder::new().add_default_ops().build().await;
        const K: usize = 64;
        const M: usize = 32;
        const N: usize = 48;

        // create test data for matrix A (K x M) as f16
        let a_data: Arc<_> = (0..M)
            .cartesian_product(0..K)
            .map(|(m, k)| fastrand::f32() * 0.1 + m as f32 * 0.01 + k as f32 * 0.001)
            .map(f16::from_f32)
            .collect();
        let a_tensor = Tensor::create(cpu.clone(), [K, M], a_data.clone())?;
        let a_matrix = MatrixFp16::from_tensor(a_tensor)?;

        // create test data for matrix B (K x N) as f32 - note: this is transposed format
        let b_data: Arc<_> = (0..N)
            .cartesian_product(0..K)
            .map(|(n, k)| fastrand::f32() * 0.1 + n as f32 * 0.002 + k as f32 * 0.003)
            .collect();
        let b_tensor = Tensor::create(cpu.clone(), [K, N], b_data.clone())?;

        // perform matrix multiplication (f16 x f32 -> f32)
        let result = a_matrix.matmul(b_tensor)?;
        let output = result.back().await?;

        // compute reference result using standard matrix multiplication
        // A is K x M (f16), B is K x N (f32), result should be M x N (f32)
        let mut r#ref = vec![0.0f32; M * N];
        for (n, m) in itertools::iproduct!(0..N, 0..M) {
            let mut sum = 0.0f32;
            for k in 0..K {
                let a_val = f16::to_f32(a_data[m * K + k]);
                let b_val = b_data[n * K + k];
                sum += a_val * b_val;
            }
            r#ref[n * M + m] = sum;
        }

        for (index, (&computed, &expected)) in output.iter().zip_eq(r#ref.iter()).enumerate() {
            assert_approx_eq!(index, computed, expected, 1e-4);
        }

        Ok(())
    }
}
