//! Build the Cube Memory shader crate to a SPIR-V binary.
//!
//! Compiles `cube-memory-shader/` against the rust-gpu compiler
//! backend. Targets `spirv-unknown-vulkan1.2` which is broadly
//! supported by current Vulkan drivers (Mesa RADV included).
//!
//! Output is a single .spv file printed to stdout on success.

use std::path::PathBuf;

use spirv_builder::{Capability, SpirvBuilder, SpirvMetadata};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Workspace-relative path to the shader crate.
    let shader_crate = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("no parent dir")?
        .join("cube-memory-shader");

    let result = SpirvBuilder::new(&shader_crate, "spirv-unknown-vulkan1.2")
        .release(true)
        .multimodule(false)
        // `Full` metadata embeds full Rust source as OpString debug
        // info — useful for shader debugging but inflates the binary
        // ~50x. `None` is the right default; switch to `Full` only
        // when you need to map a SPIR-V crash back to a source line.
        .spirv_metadata(SpirvMetadata::None)
        // Required for `subgroup_f_add` (`OpGroupNonUniformFAdd`) used
        // by cube_memory_cleanup_score and cube_memory_retrieve_score.
        // Vulkan 1.1+ devices that report subgroup-size 64 (RDNA wave64)
        // expose this; rust-gpu does not auto-infer it from the intrinsic
        // call site, so we declare it explicitly.
        .capability(Capability::GroupNonUniformArithmetic)
        .build()?;

    let spv = result.module.unwrap_single();
    println!("SPIR-V written to: {}", spv.display());
    Ok(())
}
