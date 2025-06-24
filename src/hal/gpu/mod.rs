use crate::loom::{layout::Layout, num::DataType};

/// A GPU kernel launch.
#[derive(Debug, Clone)]
pub struct Launch {
    /// The grid layout defining the number and arrangement of thread blocks.
    pub blocks: Layout,
    /// The block layout defining the number and arrangement of threads per block.
    pub threads: Layout,
    /// The kernel inputs.
    pub inputs: Vec<LayoutBuffer>,
    /// The kernel outputs.
    pub outputs: Vec<LayoutBuffer>,
}

/// A bundle of layouts that defines the mapping from GPU grid to buffer indices.
#[derive(Debug, Clone)]
pub struct LayoutBundle {
    /// The block-level layout defining how thread blocks are arranged in the grid.
    pub block: Layout,
    /// The thread-level layout defining how threads are arranged within each block.
    pub thread: Layout,
    /// Custom layout for application-specific indexing or tiling patterns.
    pub custom: Layout,
}

/// A buffer with associated layout information and data type.
#[derive(Debug, Clone)]
pub struct LayoutBuffer {
    /// The bundle of layouts defining the buffer's memory organization and access patterns.
    pub layout: LayoutBundle,
    /// The data type of elements stored in the buffer.
    pub r#type: DataType,
    /// The underlying WebGPU buffer resource.
    pub buffer: wgpu::Buffer,
}
