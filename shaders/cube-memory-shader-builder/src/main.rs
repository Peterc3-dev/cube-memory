//! Build the Cube Memory shader crate to a SPIR-V binary.
//!
//! Compiles `cube-memory-shader/` against the rust-gpu compiler
//! backend. Targets `spirv-unknown-vulkan1.2` which is broadly
//! supported by current Vulkan drivers (Mesa RADV included).
//!
//! Output is a single .spv file printed to stdout on success.

use std::path::PathBuf;

use spirv_builder::{MetadataPrintout, SpirvBuilder, SpirvMetadata};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Workspace-relative path to the shader crate.
    let shader_crate = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("no parent dir")?
        .join("cube-memory-shader");

    let result = SpirvBuilder::new(&shader_crate, "spirv-unknown-vulkan1.2")
        .release(true)
        .multimodule(false)
        .spirv_metadata(SpirvMetadata::Full)
        .print_metadata(MetadataPrintout::DependencyOnly)
        .build()?;

    let spv = result.module.unwrap_single();
    println!("SPIR-V written to: {}", spv.display());
    Ok(())
}
