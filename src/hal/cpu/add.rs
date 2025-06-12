use glam::Vec4;
use half::f16;

use crate::{
    hal::ops::AddOp,
    loom::{
        device::{Backend as _, cpu::Backend},
        num::F32x4,
        ops::{BackendOp, TensorIr},
    },
};

impl BackendOp<Backend> for AddOp<f32> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let x = backend.fetch(io[0].id);
        let y = backend.fetch(io[1].id);

        let x = x.read_slice::<f32>();
        let y = y.read_slice::<f32>();

        #[cfg(not(feature = "rayon"))]
        let output = {
            use itertools::Itertools;

            x.iter()
                .zip_eq(y.iter())
                .map(|(x, y)| x + y)
                .flat_map(|z| z.to_ne_bytes())
                .collect()
        };
        #[cfg(feature = "rayon")]
        let output = {
            use rayon::prelude::*;

            x.par_iter()
                .zip_eq(y.par_iter())
                .map(|(x, y)| x + y)
                .flat_map(|z| z.to_ne_bytes())
                .collect()
        };
        *backend.fetch(io[2].id).write() = output;
    }
}

impl BackendOp<Backend> for AddOp<f16> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let x = backend.fetch(io[0].id);
        let y = backend.fetch(io[1].id);

        let x = x.read_slice::<f16>();
        let y = y.read_slice::<f16>();

        #[cfg(not(feature = "rayon"))]
        let output = {
            use itertools::Itertools;

            x.iter()
                .zip_eq(y.iter())
                .map(|(x, y)| x + y)
                .flat_map(|z| z.to_ne_bytes())
                .collect()
        };
        #[cfg(feature = "rayon")]
        let output = {
            use rayon::prelude::*;

            x.par_iter()
                .zip_eq(y.par_iter())
                .map(|(x, y)| x + y)
                .flat_map(|z| z.to_ne_bytes())
                .collect()
        };
        *backend.fetch(io[2].id).write() = output;
    }
}

impl BackendOp<Backend> for AddOp<F32x4> {
    async fn execute(&self, backend: &mut Backend, io: Vec<TensorIr>) {
        let x = backend.fetch(io[0].id);
        let y = backend.fetch(io[1].id);

        let x = x.read_slice::<F32x4>();
        let y = y.read_slice::<F32x4>();

        #[cfg(not(feature = "rayon"))]
        let output = {
            use itertools::Itertools;

            x.iter()
                .map(|x| x.0)
                .map(Vec4::from_array)
                .zip_eq(y.iter().map(|y| y.0).map(Vec4::from_array))
                .map(|(x, y)| x + y)
                .flat_map(|z| z.to_array())
                .flat_map(|z| z.to_ne_bytes())
                .collect()
        };
        #[cfg(feature = "rayon")]
        let output = {
            use rayon::prelude::*;

            x.par_iter()
                .map(|x| x.0)
                .map(Vec4::from_array)
                .zip_eq(y.par_iter().map(|y| y.0).map(Vec4::from_array))
                .map(|(x, y)| x + y)
                .flat_map(|z| z.to_array())
                .flat_map(|z| z.to_ne_bytes())
                .collect()
        };
        *backend.fetch(io[2].id).write() = output;
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use half::f16;
    use itertools::Itertools;
    use rayon::prelude::*;

    use crate::loom::{device::CpuBuilder, tensor::Tensor};

    #[tokio::test]
    async fn test_add() -> Result<(), Box<dyn Error>> {
        let cpu = CpuBuilder::new().add_default_ops().build().await;
        const C: usize = 1024;
        const T: usize = 768;

        let data = (0..C * T).map(|x| f16::from_f32(x as f32)).collect_vec();

        let a = Tensor::create(cpu.clone(), [C, T], data.clone())?;
        let b = Tensor::create(cpu.clone(), [C, T], data.clone())?;
        let b = a.clone() + b;

        let c = Tensor::create(cpu.clone(), [C, T], data.clone())?;
        let d = a + b.clone() + c;

        let r#ref = data.par_iter().map(|x| x + x + x + x).collect::<Box<_>>();

        let output = d.back().await?;
        assert_eq!(output, r#ref);

        let d = b.clone() + b;

        let output = d.back().await?;
        assert_eq!(output, r#ref);

        Ok(())
    }
}
