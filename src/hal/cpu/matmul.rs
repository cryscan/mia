use half::f16;

use crate::{
    hal::ops::MatMatFp16Op,
    loom::{
        device::{Backend as _, cpu::Backend},
        layout::{IndexFn, Layout},
        num::F16x4,
        ops::{BackendOp, TensorIr},
        platform::handle,
    },
};

impl BackendOp<Backend> for MatMatFp16Op<F16x4> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let a = backend.fetch(io[0].id);
        let b = backend.fetch(io[1].id);

        let [layout_a, layout_b, layout_c] = self.layouts.clone();
        let [tk4, tm, bk, bm] = layout_a.shape().to_array();
        let [_, tn, _, bn] = layout_b.shape().to_array();
        assert_eq!([tm, tn, bm, bn], layout_c.shape().to_array::<4>());

        let output: Vec<_> = handle(move || {
            use itertools::Itertools;
            use rayon::prelude::*;

            itertools::iproduct!(0..bn, 0..bm)
                .collect_vec()
                .into_par_iter()
                .flat_map(move |(j, i)| {
                    let layout_a = layout_a.clone();
                    let layout_b = layout_b.clone();
                    let a = a.clone();
                    let b = b.clone();
                    (0..bk)
                        .into_par_iter()
                        .map(move |k| {
                            let layout_sa = Layout::from_shape([tk4, tm]);
                            let layout_sb = Layout::from_shape([tk4, tn]);
                            let layout_sc = Layout::from_shape([tm, tn]);
                            let mut sa = vec![F16x4::default(); layout_sa.co_size()];
                            let mut sb = vec![F16x4::default(); layout_sb.co_size()];
                            let mut sc = vec![f16::default(); layout_sc.co_size()];

                            let a = a.read_slice::<F16x4>();
                            let b = b.read_slice::<F16x4>();

                            for (y, x) in itertools::iproduct!(0..tm, 0..tk4) {
                                sa[layout_sa.value([x, y])] = a[layout_a.value([x, y, k, i])]
                            }
                            for (y, x) in itertools::iproduct!(0..tn, 0..tk4) {
                                sb[layout_sb.value([x, y])] = b[layout_b.value([x, y, k, j])];
                            }
                            for (y, x, z) in itertools::iproduct!(0..tn, 0..tm, 0..tk4) {
                                let ra = sa[layout_sa.value([x, z])];
                                let rb = sb[layout_sb.value([y, z])];
                                sc[layout_sc.value([x, y])] = ra.dot(rb);
                            }
                            sc
                        })
                        .reduce(
                            || vec![f16::default(); tm * tn],
                            |a, b| a.into_iter().zip_eq(b).map(|(a, b)| a + b).collect(),
                        )
                })
                .collect()
        })
        .await;

        backend.create(io[2].id, output);
    }
}
