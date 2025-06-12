use half::f16;

use crate::{
    hal::ops::AddOp,
    loom::{
        device::{Backend as _, cpu::Backend},
        ops::{BackendOp, TensorIr},
    },
};

impl BackendOp<Backend> for AddOp<f32> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        #[cfg(not(feature = "rayon"))]
        let output = {
            use itertools::Itertools;

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
        #[cfg(feature = "rayon")]
        let output = {
            use rayon::prelude::*;

            let x = backend.fetch(io[0].id);
            let y = backend.fetch(io[1].id);

            let (sender, receiver) = flume::bounded(0);
            crate::loom::platform::spawn_blocking(move || {
                let x = x.read_slice::<f32>();
                let y = y.read_slice::<f32>();

                let output = x
                    .par_iter()
                    .zip_eq(y.par_iter())
                    .map(|(x, y)| x + y)
                    .flat_map(|z| z.to_ne_bytes())
                    .collect();
                _ = sender.send(output);
            });
            receiver.recv_async().await.expect("failed to receive")
        };
        *backend.fetch(io[2].id).write() = output;
    }
}

impl BackendOp<Backend> for AddOp<f16> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        #[cfg(not(feature = "rayon"))]
        let output = {
            use itertools::Itertools;

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
        #[cfg(feature = "rayon")]
        let output = {
            use rayon::prelude::*;

            let x = backend.fetch(io[0].id);
            let y = backend.fetch(io[1].id);

            let (sender, receiver) = flume::bounded(0);
            crate::loom::platform::spawn_blocking(move || {
                let x = x.read_slice::<f16>();
                let y = y.read_slice::<f16>();

                let output = x
                    .par_iter()
                    .zip_eq(y.par_iter())
                    .map(|(x, y)| x + y)
                    .flat_map(|z| z.to_ne_bytes())
                    .collect();
                _ = sender.send(output);
            });
            receiver.recv_async().await.expect("failed to receive")
        };
        *backend.fetch(io[2].id).write() = output;
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use half::f16;
    use itertools::Itertools;

    use crate::loom::{device::CpuBuilder, tensor::Tensor};

    #[tokio::test]
    async fn test_add() -> Result<(), Box<dyn Error>> {
        let cpu = CpuBuilder::new().add_default_ops().build().await;

        let data = (0..12).map(|x| f16::from_f32(x as f32)).collect_vec();

        let a = Tensor::create(cpu.clone(), [4, 3], data.clone())?;
        let b = Tensor::create(cpu.clone(), [4, 3], data.clone())?;
        let b = a.clone() + b;

        let c = Tensor::create(cpu.clone(), [4, 3], data.clone())?;
        let d = a + b.clone() + c;

        let r#ref = data
            .iter()
            .map(|x| x + x + x + x)
            .collect_vec()
            .into_boxed_slice();

        let output = d.back().await?;
        assert_eq!(output, r#ref);

        let d = b.clone() + b;

        let output = d.back().await?;
        assert_eq!(output, r#ref);

        Ok(())
    }
}
