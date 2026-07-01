//! 3D Paddle Hitting Environment
//!
//! Agent controls a paddle to hit a falling ball and keep it in the air.
//! Uses wgpu for headless 3D rendering and rapier3d for physics simulation.
//!
//! Scene: red ball + blue paddle + gray ground, viewed from a fixed side angle.
//! Action: 2D continuous (paddle x-z movement).
//! Reward: hit bonus + height bonus + proximity bonus.

#![allow(dead_code, unused_variables)]

use crate::envs::Environment;
use burn::tensor::{backend::Backend, Tensor};
use rand::Rng;
use rapier3d::prelude::*;
use wgpu::util::DeviceExt;

// ── Math helpers ────────────────────────────────────────────────────

/// A simple 4x4 matrix (column-major, compatible with WGSL).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Mat4 {
    data: [[f32; 4]; 4],
}

impl Mat4 {
    fn identity() -> Self {
        let mut m = Mat4 { data: [[0.0; 4]; 4] };
        for i in 0..4 { m.data[i][i] = 1.0; }
        m
    }

    fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y / 2.0).tan();
        let mut m = Mat4 { data: [[0.0; 4]; 4] };
        m.data[0][0] = f / aspect;
        m.data[1][1] = f;
        m.data[2][2] = (far + near) / (near - far);
        m.data[2][3] = -1.0;
        m.data[3][2] = (2.0 * far * near) / (near - far);
        m
    }

    fn look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> Self {
        let f = normalize(sub(center, eye));
        let s = normalize(cross(f, up));
        let u = cross(s, f);
        let mut m = Mat4 { data: [[0.0; 4]; 4] };
        m.data[0][0] = s[0]; m.data[1][0] = s[1]; m.data[2][0] = s[2];
        m.data[0][1] = u[0]; m.data[1][1] = u[1]; m.data[2][1] = u[2];
        m.data[0][2] = -f[0]; m.data[1][2] = -f[1]; m.data[2][2] = -f[2];
        m.data[3][0] = -dot(s, eye);
        m.data[3][1] = -dot(u, eye);
        m.data[3][2] = dot(f, eye);
        m.data[3][3] = 1.0;
        m
    }

    fn translate(tx: f32, ty: f32, tz: f32) -> Self {
        let mut m = Mat4::identity();
        m.data[3][0] = tx; m.data[3][1] = ty; m.data[3][2] = tz;
        m
    }

    fn scale(sx: f32, sy: f32, sz: f32) -> Self {
        let mut m = Mat4::identity();
        m.data[0][0] = sx; m.data[1][1] = sy; m.data[2][2] = sz;
        m
    }

    fn mul(&self, other: &Mat4) -> Mat4 {
        let mut m = Mat4 { data: [[0.0; 4]; 4] };
        for col in 0..4 {
            for row in 0..4 {
                m.data[col][row] = self.data[0][row] * other.data[col][0]
                    + self.data[1][row] * other.data[col][1]
                    + self.data[2][row] * other.data[col][2]
                    + self.data[3][row] * other.data[col][3];
            }
        }
        m
    }

    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
fn len(v: [f32; 3]) -> f32 { (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt() }
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = len(v); if l > 1e-8 { [v[0]/l, v[1]/l, v[2]/l] } else { [0.0; 3] }
}

// ── Vertex type ──────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
    ];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

// ── Uniform ──────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    color: [f32; 4],
    light_dir: [f32; 3],
    _pad: f32,
}

// ── Geometry generators ──────────────────────────────────────────────

/// Generate UV sphere vertices and indices.
fn generate_sphere(radius: f32, rings: u32, sectors: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut indices = Vec::new();

    let r = rings as f32;
    let s = sectors as f32;

    for i in 0..=rings {
        let phi = std::f32::consts::PI * i as f32 / r;
        for j in 0..=sectors {
            let theta = 2.0 * std::f32::consts::PI * j as f32 / s;
            let x = phi.sin() * theta.cos();
            let y = phi.cos();
            let z = phi.sin() * theta.sin();
            verts.push(Vertex {
                position: [x * radius, y * radius, z * radius],
                normal: [x, y, z],
            });
        }
    }

    for i in 0..rings {
        for j in 0..sectors {
            let a = i * (sectors + 1) + j;
            let b = a + sectors + 1;
            indices.push(a); indices.push(b); indices.push(a + 1);
            indices.push(b); indices.push(b + 1); indices.push(a + 1);
        }
    }

    (verts, indices)
}

/// Generate box vertices and indices.
fn generate_box(hx: f32, hy: f32, hz: f32) -> (Vec<Vertex>, Vec<u32>) {
    // 6 faces × 4 vertices each, each face has its own normal
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([0.0, 1.0, 0.0], [[-hx, hy, -hz], [hx, hy, -hz], [hx, hy, hz], [-hx, hy, hz]]), // +Y
        ([0.0, -1.0, 0.0], [[-hx, -hy, hz], [hx, -hy, hz], [hx, -hy, -hz], [-hx, -hy, -hz]]), // -Y
        ([1.0, 0.0, 0.0], [[hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz], [hx, -hy, hz]]), // +X
        ([-1.0, 0.0, 0.0], [[-hx, -hy, hz], [-hx, hy, hz], [-hx, hy, -hz], [-hx, -hy, -hz]]), // -X
        ([0.0, 0.0, 1.0], [[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]]), // +Z
        ([0.0, 0.0, -1.0], [[hx, -hy, -hz], [-hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz]]), // -Z
    ];

    let mut verts = Vec::new();
    let mut indices = Vec::new();

    for (normal, corners) in &faces {
        let base = verts.len() as u32;
        for corner in corners {
            verts.push(Vertex { position: *corner, normal: *normal });
        }
        indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }

    (verts, indices)
}

/// Generate ground plane quad.
fn generate_plane(size: f32) -> (Vec<Vertex>, Vec<u32>) {
    let h = size / 2.0;
    let n = [0.0, 1.0, 0.0];
    let verts = vec![
        Vertex { position: [-h, 0.0, -h], normal: n },
        Vertex { position: [h, 0.0, -h], normal: n },
        Vertex { position: [h, 0.0, h], normal: n },
        Vertex { position: [-h, 0.0, h], normal: n },
    ];
    let indices = vec![0, 1, 2, 0, 2, 3];
    (verts, indices)
}

// ── WGPU helpers ─────────────────────────────────────────────────────

fn create_vertex_buffer(device: &wgpu::Device, verts: &[Vertex]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vertex buffer"),
        contents: bytemuck::cast_slice(verts),
        usage: wgpu::BufferUsages::VERTEX,
    })
}

fn create_index_buffer(device: &wgpu::Device, indices: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("index buffer"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    })
}

// ── WGPU pipeline ────────────────────────────────────────────────────

const SHADER_SRC: &str = r#"
struct Uniforms {
    mvp: mat4x4<f32>,
    color: vec4<f32>,
    light_dir: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvp * vec4<f32>(in.position, 1.0);
    out.world_normal = in.normal;
    out.world_pos = in.position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ambient = 0.25;
    let n = normalize(in.world_normal);
    let l = normalize(uniforms.light_dir);
    let diffuse = max(dot(n, l), 0.0) * 0.75;
    let lit = ambient + diffuse;
    return vec4<f32>(uniforms.color.rgb * lit, 1.0);
}
"#;

struct RenderObjects {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    // ball
    ball_vb: wgpu::Buffer,
    ball_ib: wgpu::Buffer,
    ball_index_count: u32,
    // paddle
    paddle_vb: wgpu::Buffer,
    paddle_ib: wgpu::Buffer,
    paddle_index_count: u32,
    // ground
    ground_vb: wgpu::Buffer,
    ground_ib: wgpu::Buffer,
    ground_index_count: u32,
    // uniform
    uniform_buf: wgpu::Buffer,
    // depth
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    // output
    output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,
    output_buf: wgpu::Buffer,
    // bind groups per object
    ball_bind_group: wgpu::BindGroup,
    paddle_bind_group: wgpu::BindGroup,
    ground_bind_group: wgpu::BindGroup,
    image_size: u32,
    padded_bytes_per_row: u32,
}

impl RenderObjects {
    fn new(device: &wgpu::Device, image_size: u32) -> Self {
        // Geometry
        let (ball_v, ball_i) = generate_sphere(0.08, 16, 16);
        let (paddle_v, paddle_i) = generate_box(0.25, 0.03, 0.18);
        let (ground_v, ground_i) = generate_plane(2.0);

        let ball_vb = create_vertex_buffer(device, &ball_v);
        let ball_ib = create_index_buffer(device, &ball_i);
        let ball_ic = ball_i.len() as u32;
        let paddle_vb = create_vertex_buffer(device, &paddle_v);
        let paddle_ib = create_index_buffer(device, &paddle_i);
        let paddle_ic = paddle_i.len() as u32;
        let ground_vb = create_vertex_buffer(device, &ground_v);
        let ground_ib = create_index_buffer(device, &ground_i);
        let ground_ic = ground_i.len() as u32;

        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Uniform buffer
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Helper to make bind group for a specific color
        let make_bg = |color: [f32; 4]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bind group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                }],
            })
        };

        let ball_bind_group = make_bg([0.9, 0.2, 0.15, 1.0]);  // red
        let paddle_bind_group = make_bg([0.2, 0.4, 0.9, 1.0]);  // blue
        let ground_bind_group = make_bg([0.3, 0.3, 0.35, 1.0]); // gray

        // Depth
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width: image_size, height: image_size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Output texture
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("output"),
            size: wgpu::Extent3d { width: image_size, height: image_size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Output buffer (for CPU readback)
        let bytes_per_row = ((image_size * 4) as f64 / 256.0).ceil() as u32 * 256;
        let padded_bytes_per_row = bytes_per_row as usize;
        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output buf"),
            size: (padded_bytes_per_row * image_size as usize) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        RenderObjects {
            pipeline,
            bind_group_layout,
            ball_vb, ball_ib, ball_index_count: ball_ic,
            paddle_vb, paddle_ib, paddle_index_count: paddle_ic,
            ground_vb, ground_ib, ground_index_count: ground_ic,
            uniform_buf,
            depth_texture, depth_view,
            output_texture, output_view, output_buf,
            ball_bind_group, paddle_bind_group, ground_bind_group,
            image_size,
            padded_bytes_per_row: bytes_per_row,
        }
    }

    fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_proj: &Mat4,
        ball_pos: [f32; 3],
        paddle_pos: [f32; 3],
    ) -> Vec<f32> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render encoder"),
        });

        // Write ball uniform
        let ball_mvp = view_proj.mul(&Mat4::translate(ball_pos[0], ball_pos[1], ball_pos[2]));
        let u_ball = Uniforms {
            mvp: ball_mvp.data,
            color: [0.9, 0.2, 0.15, 1.0],
            light_dir: normalize([1.0, 2.0, 1.5]),
            _pad: 0.0,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u_ball));

        // Render pass
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05, g: 0.05, b: 0.1, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.pipeline);

            // Draw ball
            rpass.set_bind_group(0, &self.ball_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.ball_vb.slice(..));
            rpass.set_index_buffer(self.ball_ib.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..self.ball_index_count, 0, 0..1);

            // Draw paddle
            let paddle_mvp = view_proj.mul(&Mat4::translate(paddle_pos[0], paddle_pos[1], paddle_pos[2]));
            let u_paddle = Uniforms {
                mvp: paddle_mvp.data,
                color: [0.2, 0.4, 0.9, 1.0],
                light_dir: normalize([1.0, 2.0, 1.5]),
                _pad: 0.0,
            };
            queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u_paddle));
            rpass.set_bind_group(0, &self.paddle_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.paddle_vb.slice(..));
            rpass.set_index_buffer(self.paddle_ib.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..self.paddle_index_count, 0, 0..1);

            // Draw ground
            let u_ground = Uniforms {
                mvp: view_proj.data,
                color: [0.3, 0.3, 0.35, 1.0],
                light_dir: normalize([1.0, 2.0, 1.5]),
                _pad: 0.0,
            };
            queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u_ground));
            rpass.set_bind_group(0, &self.ground_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.ground_vb.slice(..));
            rpass.set_index_buffer(self.ground_ib.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..self.ground_index_count, 0, 0..1);
        }

        // Copy texture to buffer (with aligned row stride)
        let size = self.image_size;
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.output_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        );

        queue.submit(Some(encoder.finish()));

        // Read back pixels (skip padding bytes per row)
        let buf_slice = self.output_buf.slice(..);
        buf_slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = buf_slice.get_mapped_range();
        let raw: Vec<u8> = data.to_vec();
        drop(data);
        self.output_buf.unmap();

        // Convert RGBA (with padded stride) → RGB f32 (CHW order)
        let stride = self.padded_bytes_per_row as usize;
        let unpadded = (size * 4) as usize;
        let mut rgb = vec![0.0f32; (3 * size * size) as usize];
        for y in 0..size {
            for x in 0..size {
                let src_idx = y as usize * stride + x as usize * 4;
                let r = raw[src_idx] as f32 / 255.0;
                let g = raw[src_idx + 1] as f32 / 255.0;
                let b = raw[src_idx + 2] as f32 / 255.0;
                let dst_offset = (y * size + x) as usize;
                rgb[dst_offset] = r;
                rgb[(size*size) as usize + dst_offset] = g;
                rgb[2*(size*size) as usize + dst_offset] = b;
            }
        }
        rgb
    }
}

// ── Camera ───────────────────────────────────────────────────────────

struct Camera {
    eye: [f32; 3],
    center: [f32; 3],
    up: [f32; 3],
    fov_y: f32,
    near: f32,
    far: f32,
}

impl Camera {
    fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at(self.eye, self.center, self.up);
        let proj = Mat4::perspective(self.fov_y, aspect, self.near, self.far);
        proj.mul(&view)
    }
}

// ── PaddleHitting Environment ────────────────────────────────────────

pub struct PaddleHitting {
    // Render
    device: wgpu::Device,
    queue: wgpu::Queue,
    render_objs: RenderObjects,
    camera: Camera,

    // Physics
    physics_pipeline: PhysicsPipeline,
    gravity: Vector<f32>,
    integration_params: IntegrationParameters,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    query_pipeline: QueryPipeline,
    ball_handle: RigidBodyHandle,
    paddle_handle: RigidBodyHandle,

    // Episode
    step_count: usize,
    max_steps: usize,
    obs_shape: [usize; 3],
    hits: u32,

    // RNG
    rng: rand::rngs::StdRng,
}

impl PaddleHitting {
    pub fn new(
        max_steps: usize,
        action_dim: usize,
        image_channels: usize,
        image_size: usize,
        seed: u64,
    ) -> Self {
        use rand::SeedableRng;
        let rng = rand::rngs::StdRng::seed_from_u64(seed);

        // ── Init wgpu ──
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        })).expect("Failed to get wgpu adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("PaddleHitting device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        )).expect("Failed to create wgpu device");

        let render_objs = RenderObjects::new(&device, image_size as u32);

        let camera = Camera {
            eye: [1.0, 1.0, 1.2],
            center: [0.5, 0.3, 0.5],
            up: [0.0, 1.0, 0.0],
            fov_y: 60.0_f32.to_radians(),
            near: 0.1,
            far: 10.0,
        };

        // ── Init rapier3d ──
        let mut rigid_body_set = RigidBodySet::new();
        let mut collider_set = ColliderSet::new();

        // Ball: dynamic, sphere
        let ball_rb = RigidBodyBuilder::dynamic()
            .translation(vector![0.5, 0.8, 0.5])
            .linvel(vector![0.0, -0.5, 0.3])
            .build();
        let ball_handle = rigid_body_set.insert(ball_rb);
        let ball_collider = ColliderBuilder::ball(0.08)
            .restitution(0.7)
            .build();
        collider_set.insert_with_parent(ball_collider, ball_handle, &mut rigid_body_set);

        // Paddle: kinematic, cuboid
        let paddle_rb = RigidBodyBuilder::kinematic_position_based()
            .translation(vector![0.5, 0.12, 0.5])
            .build();
        let paddle_handle = rigid_body_set.insert(paddle_rb);
        let paddle_collider = ColliderBuilder::cuboid(0.25, 0.03, 0.18)
            .restitution(0.8)
            .build();
        collider_set.insert_with_parent(paddle_collider, paddle_handle, &mut rigid_body_set);

        // Ground: static, cuboid
        let ground_rb = RigidBodyBuilder::fixed()
            .translation(vector![0.5, -0.05, 0.5])
            .build();
        let ground_handle = rigid_body_set.insert(ground_rb);
        let ground_collider = ColliderBuilder::cuboid(1.0, 0.05, 1.0)
            .build();
        collider_set.insert_with_parent(ground_collider, ground_handle, &mut rigid_body_set);

        // Walls (4 sides)
        let wall_thickness = 0.05;
        let walls = [
            // +X
            (vector![1.0 + wall_thickness/2.0, 0.5, 0.5], vector![wall_thickness/2.0, 0.5, 0.5]),
            // -X
            (vector![-wall_thickness/2.0, 0.5, 0.5], vector![wall_thickness/2.0, 0.5, 0.5]),
            // +Z
            (vector![0.5, 0.5, 1.0 + wall_thickness/2.0], vector![0.5, 0.5, wall_thickness/2.0]),
            // -Z
            (vector![0.5, 0.5, -wall_thickness/2.0], vector![0.5, 0.5, wall_thickness/2.0]),
        ];
        for (pos, half_ext) in &walls {
            let wall_rb = RigidBodyBuilder::fixed().translation(*pos).build();
            let wh = rigid_body_set.insert(wall_rb);
            let wc = ColliderBuilder::cuboid(half_ext.x, half_ext.y, half_ext.z)
                .restitution(0.6)
                .build();
            collider_set.insert_with_parent(wc, wh, &mut rigid_body_set);
        }

        Self {
            device, queue, render_objs, camera,
            physics_pipeline: PhysicsPipeline::new(),
            gravity: vector![0.0, -9.81, 0.0],
            integration_params: IntegrationParameters::default(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            rigid_body_set, collider_set,
            query_pipeline: QueryPipeline::new(),
            ball_handle, paddle_handle,
            step_count: 0,
            max_steps,
            obs_shape: [image_channels, image_size, image_size],
            hits: 0,
            rng,
        }
    }

    fn render_frame<B: Backend>(&self, device: &B::Device) -> Tensor<B, 3> {
        let ball_pos = {
            let rb = self.rigid_body_set.get(self.ball_handle).unwrap();
            let t = rb.translation();
            [t.x, t.y, t.z]
        };
        let paddle_pos = {
            let rb = self.rigid_body_set.get(self.paddle_handle).unwrap();
            let t = rb.translation();
            [t.x, t.y, t.z]
        };

        let aspect = 1.0_f32;
        let vp = self.camera.view_proj(aspect);
        let pixels = self.render_objs.render(&self.device, &self.queue, &vp, ball_pos, paddle_pos);

        let [c, h, w] = self.obs_shape;
        Tensor::<B, 1>::from_floats(pixels.as_slice(), device).reshape([c, h, w])
    }

    fn physics_step(&mut self) {
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_params,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            None,
            &(),
            &(),
        );

        // Update query pipeline
        self.query_pipeline.update(&self.collider_set);
    }

    fn set_paddle_position(&mut self, fx: f32, fz: f32) {
        let paddle = self.rigid_body_set.get_mut(self.paddle_handle).unwrap();

        // Move paddle in x-z plane, bounded
        let max_move = 0.08;
        let mut new_x = paddle.translation().x + fx * max_move;
        let mut new_z = paddle.translation().z + fz * max_move;

        // Keep paddle bounds: paddle is at y=0.12, size 0.25x0.18
        // So x ∈ [0.25, 0.75], z ∈ [0.23, 0.77] (with 0.05 wall on each side)
        new_x = new_x.clamp(0.25, 0.75);
        new_z = new_z.clamp(0.23, 0.77);

        paddle.set_translation(vector![new_x, 0.12, new_z], true);
    }

    fn compute_reward(&self, had_collision: bool) -> f32 {
        let ball = self.rigid_body_set.get(self.ball_handle).unwrap();
        let ball_pos = ball.translation();
        let paddle = self.rigid_body_set.get(self.paddle_handle).unwrap();
        let paddle_pos = paddle.translation();

        let height_reward = (ball_pos.y * 3.0).max(0.0);

        // Proximity reward: encourage paddle to stay under the ball
        let dx = ball_pos.x - paddle_pos.x;
        let dz = ball_pos.z - paddle_pos.z;
        let dist = (dx * dx + dz * dz).sqrt();
        let proximity_reward = (1.0 - dist).max(0.0) * 0.5;

        let hit_bonus = if had_collision { 2.0 } else { 0.0 };

        height_reward + proximity_reward + hit_bonus
    }

    fn check_ball_paddle_collision(&self) -> bool {
        // Simple distance-based collision check between ball and paddle
        let ball = self.rigid_body_set.get(self.ball_handle).unwrap();
        let paddle = self.rigid_body_set.get(self.paddle_handle).unwrap();
        let bp = ball.translation();
        let pp = paddle.translation();

        let dx = bp.x - pp.x;
        let dy = bp.y - pp.y;
        let dz = bp.z - pp.z;

        // Approximate: ball radius 0.08 + paddle half-height 0.03 ~= 0.11
        let contact_dist = 0.11;
        let paddle_half_x = 0.25;
        let paddle_half_z = 0.18;

        dx.abs() < paddle_half_x + 0.08
            && dz.abs() < paddle_half_z + 0.08
            && dy < contact_dist
            && dy > -0.05
    }
}

impl Environment for PaddleHitting {
    fn obs_shape(&self) -> [usize; 3] {
        self.obs_shape
    }

    fn action_dim(&self) -> usize {
        2 // x-z paddle movement
    }

    fn max_steps(&self) -> usize {
        self.max_steps
    }

    fn reset<B: Backend>(&mut self, device: &B::Device) -> Tensor<B, 3> {
        self.step_count = 0;
        self.hits = 0;

        // Reset ball: random position above paddle, random velocity
        let bx = self.rng.gen_range(0.2..0.8);
        let bz = self.rng.gen_range(0.2..0.8);
        let vx = self.rng.gen_range(-1.0..1.0);
        let vz = self.rng.gen_range(-1.0..1.0);
        let ball = self.rigid_body_set.get_mut(self.ball_handle).unwrap();
        ball.set_translation(vector![bx, 0.7, bz], true);
        ball.set_linvel(vector![vx * 0.5, -0.5, vz * 0.5], true);

        // Reset paddle to center
        let paddle = self.rigid_body_set.get_mut(self.paddle_handle).unwrap();
        paddle.set_translation(vector![0.5, 0.12, 0.5], true);

        self.render_frame::<B>(device)
    }

    fn step<B: Backend>(
        &mut self,
        action: &[f32],
        device: &B::Device,
    ) -> (Tensor<B, 3>, f32, bool) {
        let fx = action[0].clamp(-1.0, 1.0);
        let fz = if action.len() > 1 { action[1].clamp(-1.0, 1.0) } else { 0.0 };

        self.set_paddle_position(fx, fz);
        self.physics_step();

        let had_collision = self.check_ball_paddle_collision();
        if had_collision {
            self.hits += 1;
        }

        self.step_count += 1;

        let ball = self.rigid_body_set.get(self.ball_handle).unwrap();
        let ball_y = ball.translation().y;
        let done = self.step_count >= self.max_steps || ball_y < -0.5;

        let reward = self.compute_reward(had_collision);

        let obs = self.render_frame::<B>(device);
        (obs, reward, done)
    }
}
