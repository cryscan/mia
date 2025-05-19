use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId;

pub trait Device {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cpu;

impl Device for Cpu {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gpu {
    /// The unique identifier of the device.
    id: uid::Id<DeviceId>,
    /// Handle to a WebGPU compute device.
    device: wgpu::Device,
    /// The WebGPU command queue.
    queue: wgpu::Queue,
}

impl Device for Gpu {}
