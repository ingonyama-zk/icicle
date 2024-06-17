pub mod device;
pub mod errors;
pub mod memory;
pub mod runtime;
pub mod stream;
pub mod tests;

// Re-export the types for easier access
pub use device::{Device, DeviceProperties};
pub use errors::eIcicleError;
pub use runtime::*;
