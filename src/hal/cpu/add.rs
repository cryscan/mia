use half::f16;

use crate::{
    hal::ops::AddOp,
    loom::{
        device::{Backend as _, cpu::Backend},
        num::{F16x4, F32x4},
        ops::{BackendOp, TensorIr},
        platform::handle,
    },
};

impl BackendOp<Backend> for AddOp<f32> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        debug_assert_eq!(io[0].layout, io[1].layout);
        let layout = io[0].layout.clone();

        let x = backend.fetch(io[0].id);
        let y = backend.fetch(io[1].id);

        #[cfg(not(feature = "rayon"))]
        let output: Vec<_> = handle(move || {
            let x = x.read_slice::<f32>();
            let y = y.read_slice::<f32>();

            layout
                .iter_indices()
                .map(|(_, index)| x[index] + y[index])
                .collect()
        })
        .await;
        #[cfg(feature = "rayon")]
        let output: Vec<_> = handle(move || {
            use rayon::prelude::*;

            let x = x.read_slice::<f32>();
            let y = y.read_slice::<f32>();

            layout
                .par_iter_indices()
                .map(|(_, index)| x[index] + y[index])
                .collect()
        })
        .await;

        backend.create(io[2].id, output);
    }
}

impl BackendOp<Backend> for AddOp<f16> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        debug_assert_eq!(io[0].layout, io[1].layout);
        let layout = io[0].layout.clone();

        let x = backend.fetch(io[0].id);
        let y = backend.fetch(io[1].id);

        #[cfg(not(feature = "rayon"))]
        let output: Vec<_> = handle(move || {
            let x = x.read_slice::<f16>();
            let y = y.read_slice::<f16>();

            layout
                .iter_indices()
                .map(|(_, index)| x[index] + y[index])
                .collect()
        })
        .await;
        #[cfg(feature = "rayon")]
        let output: Vec<_> = handle(move || {
            use rayon::prelude::*;

            let x = x.read_slice::<f16>();
            let y = y.read_slice::<f16>();

            layout
                .par_iter_indices()
                .map(|(_, index)| x[index] + y[index])
                .collect()
        })
        .await;

        backend.create(io[2].id, output);
    }
}

impl BackendOp<Backend> for AddOp<F32x4> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        debug_assert_eq!(io[0].layout, io[1].layout);
        let layout = io[0].layout.clone();

        let x = backend.fetch(io[0].id);
        let y = backend.fetch(io[1].id);

        #[cfg(not(feature = "rayon"))]
        let output: Vec<_> = handle(move || {
            let x = x.read_slice::<F32x4>();
            let y = y.read_slice::<F32x4>();

            layout
                .iter_indices()
                .map(|(_, index)| x[index] + y[index])
                .collect()
        })
        .await;
        #[cfg(feature = "rayon")]
        let output: Vec<_> = handle(move || {
            use rayon::prelude::*;

            let x = x.read_slice::<F32x4>();
            let y = y.read_slice::<F32x4>();

            layout
                .par_iter_indices()
                .map(|(_, index)| x[index] + y[index])
                .collect()
        })
        .await;

        backend.create(io[2].id, output);
    }
}

impl BackendOp<Backend> for AddOp<F16x4> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        debug_assert_eq!(io[0].layout, io[1].layout);
        let layout = io[0].layout.clone();

        let x = backend.fetch(io[0].id);
        let y = backend.fetch(io[1].id);

        #[cfg(not(feature = "rayon"))]
        let output: Vec<_> = handle(move || {
            let x = x.read_slice::<F16x4>();
            let y = y.read_slice::<F16x4>();

            layout
                .iter_indices()
                .map(|(_, index)| x[index] + y[index])
                .collect()
        })
        .await;
        #[cfg(feature = "rayon")]
        let output: Vec<_> = handle(move || {
            use rayon::prelude::*;

            let x = x.read_slice::<F16x4>();
            let y = y.read_slice::<F16x4>();

            layout
                .par_iter_indices()
                .map(|(_, index)| x[index] + y[index])
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
    use rayon::prelude::*;

    use crate::loom::{device::CpuBuilder, num::F32x4, tensor::Tensor};

    #[tokio::test]
    async fn test_add_f16() -> Result<(), Box<dyn Error>> {
        let cpu = CpuBuilder::new().add_default_ops().build().await;
        const C: usize = 1024;
        const T: usize = 768;

        let data: Arc<[f16]> = (0..C * T).map(|x| f16::from_f32(x as f32)).collect();

        let a = Tensor::create(cpu.clone(), [C, T], data.clone());
        let b = Tensor::create(cpu.clone(), [C, T], data.clone());
        let b = a.clone() + b;

        let c = Tensor::create(cpu.clone(), [C, T], data.clone());
        let d = a + b.clone() + c;

        let r#ref = data.par_iter().map(|x| x + x + x + x).collect::<Box<_>>();

        println!("{}", d.clone().mermaid().await?);
        let output = d.back().await?;
        assert_eq!(output, r#ref);

        let d = b.clone() + b;

        let output = d.back().await?;
        assert_eq!(output, r#ref);

        Ok(())
    }

    #[tokio::test]
    async fn test_add_f32x4() -> Result<(), Box<dyn Error>> {
        let cpu = CpuBuilder::new().add_default_ops().build().await;
        const C: usize = 1024;
        const T: usize = 768;

        let data: Arc<[f32]> = (0..C * T).map(|x| x as f32).collect();

        let a = Tensor::create(cpu.clone(), [C, T], data.clone()).cast::<F32x4>([C / 4, T]);
        let b = Tensor::create(cpu.clone(), [C, T], data.clone()).cast::<F32x4>([C / 4, T]);
        let b = a.clone() + b;

        let c = Tensor::create(cpu.clone(), [C, T], data.clone()).cast::<F32x4>([C / 4, T]);
        let d = a + b.clone() + c;

        let r#ref = data.par_iter().map(|x| x + x + x + x).collect::<Box<_>>();

        let output = d.cast::<f32>([C, T]).back().await?;
        assert_eq!(output, r#ref);

        let d = b.clone() + b;

        println!("{}", d.clone().mermaid().await?);
        let output = d.cast::<f32>([C, T]).back().await?;
        assert_eq!(output, r#ref);

        Ok(())
    }
}
