pub mod config;
pub mod device;
pub mod errors;
pub mod memory;
pub mod runtime;
pub mod stream;

#[doc(hidden)]
pub mod test_utilities;
pub mod tests;

// Re-export the types for easier access
pub use device::{Device, DeviceProperties};
pub use errors::{eIcicleError, IcicleError};
pub use runtime::*;
