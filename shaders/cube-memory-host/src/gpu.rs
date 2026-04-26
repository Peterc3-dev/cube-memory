//! Minimal wgpu compute harness for running the Cube Memory shaders.
//!
//! The harness intentionally re-creates the device, pipeline, and
//! buffers per kernel invocation. It is built for *correctness
//! testing*, not throughput. A production embedding (in ggml-vulkan)
//! will reuse a long-lived device + pipeline cache and pre-allocate
//! buffers — see SPEC.md Phase 2.

use std::borrow::Cow;
use std::path::Path;

use bytemuck::{NoUninit, Pod};

/// Minimal compute context: a single wgpu device + queue + a SPIR-V
/// module loaded once at construction.
pub struct GpuCtx {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub module: wgpu::ShaderModule,
}

impl GpuCtx {
    /// Initialize a headless Vulkan-backed wgpu device and load the
    /// shader module from `spv_path` (a path to the SPIR-V file
    /// produced by `cube-memory-shader-builder`).
    pub fn new(spv_path: &Path) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .expect("no wgpu Vulkan adapter found");
        let mut limits = wgpu::Limits::default();
        // Each kernel uses a small push-constant block (12 bytes max
        // across our six entry points). 32 is a safe ceiling.
        limits.max_push_constant_size = 32;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("cube-memory-host"),
                required_features: wgpu::Features::SPIRV_SHADER_PASSTHROUGH
                    | wgpu::Features::PUSH_CONSTANTS,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("device request failed");

        let bytes = std::fs::read(spv_path).expect("read SPIR-V");
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // SAFETY: spirv-val passes on this binary in the build pipeline,
        // so the module is well-formed Vulkan SPIR-V. wgpu does not
        // re-validate when using passthrough.
        let module = unsafe {
            device.create_shader_module_spirv(&wgpu::ShaderModuleDescriptorSpirV {
                label: Some("cube_memory_shader"),
                source: Cow::Owned(words),
            })
        };

        Self { device, queue, module }
    }

    /// Run a compute pipeline with a single push-constant block and
    /// `n_storage` storage buffers (descriptor_set 0, bindings 0..n).
    /// Returns the contents of the final binding (assumed output).
    ///
    /// This intentionally has the simplest possible interface — a
    /// real engine would expose finer control, but for parity tests
    /// the goal is "give me back the bytes the shader wrote to the
    /// last binding."
    pub fn run<P, T>(
        &self,
        entry_point: &str,
        push: P,
        inputs: &[&[u8]],
        out_bytes: usize,
        groups: (u32, u32, u32),
    ) -> Vec<T>
    where
        P: NoUninit,
        T: Pod + Copy,
    {
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..inputs.len() as u32 + 1)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: i < inputs.len() as u32,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();

        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cube-mem bgl"),
            entries: &bgl_entries,
        });

        let pcr = wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..std::mem::size_of::<P>() as u32,
        };

        let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cube-mem pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[pcr],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pl),
            module: &self.module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        // Input buffers + output buffer + readback.
        let in_bufs: Vec<wgpu::Buffer> = inputs
            .iter()
            .enumerate()
            .map(|(i, data)| {
                use wgpu::util::DeviceExt;
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("in_{i}")),
                    contents: data,
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .collect();
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: out_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: out_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut bg_entries: Vec<wgpu::BindGroupEntry> = in_bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        bg_entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32,
            resource: out_buf.as_entire_binding(),
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cube-mem bg"),
            layout: &bgl,
            entries: &bg_entries,
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(&pipeline);
            cp.set_bind_group(0, &bg, &[]);
            cp.set_push_constants(0, bytemuck::bytes_of(&push));
            cp.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_bytes as u64);
        self.queue.submit([enc.finish()]);

        let slice = read_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("buffer map failed");
        let data = slice.get_mapped_range();
        let out: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        read_buf.unmap();
        out
    }

}
