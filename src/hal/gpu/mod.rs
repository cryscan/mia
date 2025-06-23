use crate::loom::layout::Layout;

/// A GPU kernel.
///
/// A kernel is a set of instructions that will be executed on the GPU. It is
/// defined by a shader code and a layout that specifies the number of workgroups
/// and threads within each workgroup.
#[derive(Debug, Clone)]
pub struct Kernel {
    /// The layout of the workgroups.
    pub blocks: Layout,
    /// The layout of the threads within each workgroup.
    pub threads: Layout,
}
