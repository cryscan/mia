use std::any::Any;

use derive_more::Display;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, Serialize, Deserialize)]
pub enum TensorAccess {
    ReadOnly,
    ReadWrite,
    WriteOnly,
}

pub trait TensorOp: Send + Sync + Any {}
