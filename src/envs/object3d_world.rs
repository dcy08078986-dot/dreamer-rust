#![allow(dead_code, unused_variables)]
//! 3D Object-Centric Physics Environment for World Model Benchmarking
//!
//! Features:
//! - Multiple balls and cubes with 3D physics (gravity, collisions, friction)
//! - Fixed third-person camera rendering to 128×128 RGB
//! - Ground truth state output (position, velocity, type, size)
//! - Randomized initial conditions per episode

use crate::envs::Environment;
use burn::tensor::{backend::Backend, Tensor};

// ── 3D Vector Math ──

#[derive(Clone, Copy, Debug)]
struct Vec3(f32, f32, f32);

impl Vec3 {
    fn dot(&self, o: &Vec3) -> f32 { self.0*o.0 + self.1*o.1 + self.2*o.2 }
    fn len(&self) -> f32 { self.dot(self).sqrt() }
    fn norm(&self) -> Vec3 { let l=self.len(); if l>1e-8 {Vec3(self.0/l,self.1/l,self.2/l)} else {Vec3(0.,0.,0.)} }
    fn cross(&self, o: &Vec3) -> Vec3 { Vec3(self.1*o.2-self.2*o.1, self.2*o.0-self.0*o.2, self.0*o.1-self.1*o.0) }
}

use std::ops::{Add, Sub, Mul};
impl Add for Vec3 { type Output=Vec3; fn add(self, o:Vec3)->Vec3 { Vec3(self.0+o.0,self.1+o.1,self.2+o.2) } }
impl Sub for Vec3 { type Output=Vec3; fn sub(self, o:Vec3)->Vec3 { Vec3(self.0-o.0,self.1-o.1,self.2-o.2) } }
impl Mul<f32> for Vec3 { type Output=Vec3; fn mul(self, s:f32)->Vec3 { Vec3(self.0*s,self.1*s,self.2*s) } }

// ── Simple RNG ──

struct XorShift(u64);
impl XorShift {
    fn new(seed: u64) -> Self { XorShift(seed.wrapping_add(0x9E3779B97F4A7C15)) }
    fn next(&mut self) -> u64 { let mut x=self.0; x^=x<<13; x^=x>>7; x^=x<<17; self.0=x; x }
    fn f32(&mut self) -> f32 { (self.next() as f32)/(u64::MAX as f32) }
    fn vec3(&mut self, s:f32) -> Vec3 { Vec3((self.f32()-0.5)*s,(self.f32()-0.5)*s,(self.f32()-0.5)*s) }
}

// ── 3D Object Types ──

#[derive(Clone, Debug)]
pub enum ObjType { Ball { radius: f32 }, Cube { half_size: f32 } }

#[derive(Clone, Debug)]
pub struct Object3D {
    pub id: usize,
    pub obj_type: ObjType,
    pub pos: Vec3,
    pub vel: Vec3,
    pub mass: f32,
    pub color: [u8; 3],
}

#[derive(Clone, Debug)]
pub struct Object3DState {
    pub id: usize,
    pub type_name: String, // "ball" or "cube"
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub size: f32,
    pub color: [u8; 3],
}

#[derive(Clone, Debug)]
pub struct World3DState {
    pub objects: Vec<Object3DState>,
    pub time: usize,
}

// ── 3D Environment ──

pub struct Object3DWorld {
    pub objects: Vec<Object3D>,
    // World bounds
    world_min: f32, world_max: f32,
    // Physics
    gravity: f32, damping: f32, bounce_coef: f32,
    // Episode
    step_count: usize, max_steps: usize,
    // Camera (fixed third-person, looking at center from front-top-right)
    cam_pos: Vec3, cam_target: Vec3, cam_up: Vec3,
    // Rendering
    image_size: usize, image_channels: usize,
    // Lighting
    light_dir: Vec3,
    // RNG
    rng: XorShift,
    id_counter: usize,
}

impl Object3DWorld {
    pub fn new(
        max_steps: usize, action_dim: usize, image_channels: usize, image_size: usize,
        seed: u64, num_objects: usize,
    ) -> Self {
        let rng = XorShift::new(seed);
        let mut obj = Self {
            objects: Vec::new(),
            world_min: 0.0, world_max: 100.0,
            gravity: 0.3, damping: 0.995, bounce_coef: 0.7,
            step_count: 0, max_steps,
            cam_pos: Vec3(60.0, 80.0, 70.0),
            cam_target: Vec3(50.0, 40.0, 50.0),
            cam_up: Vec3(0.0, 1.0, 0.0),
            image_size, image_channels,
            light_dir: Vec3(0.5, -1.0, -0.3).norm(),
            rng, id_counter: 0,
        };
        obj.init_scene(num_objects);
        obj
    }

    fn init_scene(&mut self, num_objects: usize) {
        self.objects.clear();
        self.id_counter = 0;
        let colors: [[u8;3]; 7] = [
            [255,80,80],[80,180,255],[80,255,120],[255,220,60],[200,80,255],[255,140,40],[60,220,220]
        ];
        for i in 0..num_objects {
            let is_ball = i % 2 == 0 || self.rng.f32() < 0.6;
            let (obj_type, size) = if is_ball {
                (ObjType::Ball { radius: 3.0 + self.rng.f32() * 4.0 }, 3.0 + self.rng.f32() * 4.0)
            } else {
                (ObjType::Cube { half_size: 2.5 + self.rng.f32() * 3.0 }, 2.5 + self.rng.f32() * 3.0)
            };
            let mass = size * size * size * 0.01;
            // Place without overlapping
            let pos = loop {
                let p = Vec3(
                    15.0 + self.rng.f32() * 70.0,
                    10.0 + self.rng.f32() * 60.0,
                    15.0 + self.rng.f32() * 70.0,
                );
                if self.objects.iter().all(|o| {
                    let d = (p - o.pos).len();
                    let min_d = self.obj_radius(&obj_type) + self.obj_radius(&o.obj_type) + 3.0;
                    d >= min_d
                }) { break p; }
            };
            self.objects.push(Object3D {
                id: self.id_counter,
                obj_type, pos,
                vel: self.rng.vec3(2.0),
                mass,
                color: colors[i % colors.len()],
            });
            self.id_counter += 1;
        }
    }

    fn obj_radius(&self, t: &ObjType) -> f32 {
        match t { ObjType::Ball{radius} => *radius, ObjType::Cube{half_size} => *half_size * 1.5 }
    }

    pub fn state(&self) -> World3DState {
        World3DState {
            objects: self.objects.iter().map(|o| {
                let (tn, sz) = match &o.obj_type {
                    ObjType::Ball{radius} => ("ball".into(), *radius),
                    ObjType::Cube{half_size} => ("cube".into(), *half_size),
                };
                Object3DState {
                    id: o.id, type_name: tn,
                    position: [o.pos.0, o.pos.1, o.pos.2],
                    velocity: [o.vel.0, o.vel.1, o.vel.2],
                    size: sz, color: o.color,
                }
            }).collect(),
            time: self.step_count,
        }
    }

    fn physics_step(&mut self, action: f32) {
        // Apply control to first object (player)
        if !self.objects.is_empty() {
            self.objects[0].vel = self.objects[0].vel + Vec3(action * 0.5, 0.0, 0.0);
        }
        // Gravity + damping + integration
        for o in &mut self.objects {
            o.vel = o.vel + Vec3(0.0, -self.gravity, 0.0);
            o.vel = o.vel * self.damping;
            o.pos = o.pos + o.vel;
        }
        // Boundary bounce (compute radii first to avoid borrow conflict)
        let radii: Vec<f32> = self.objects.iter().map(|o| self.obj_radius(&o.obj_type)).collect();
        for (i, o) in self.objects.iter_mut().enumerate() {
            let r = radii[i];
            if o.pos.0 - r < self.world_min { o.pos.0 = self.world_min + r; o.vel.0 = o.vel.0.abs() * self.bounce_coef; }
            if o.pos.0 + r > self.world_max { o.pos.0 = self.world_max - r; o.vel.0 = -o.vel.0.abs() * self.bounce_coef; }
            if o.pos.1 - r < self.world_min { o.pos.1 = self.world_min + r; o.vel.1 = o.vel.1.abs() * self.bounce_coef; }
            if o.pos.1 + r > self.world_max { o.pos.1 = self.world_max - r; o.vel.1 = -o.vel.1.abs() * self.bounce_coef; }
            if o.pos.2 - r < self.world_min { o.pos.2 = self.world_min + r; o.vel.2 = o.vel.2.abs() * self.bounce_coef; }
            if o.pos.2 + r > self.world_max { o.pos.2 = self.world_max - r; o.vel.2 = -o.vel.2.abs() * self.bounce_coef; }
        }
        // Object-object collision (compute radii first)
        let col_radii: Vec<f32> = self.objects.iter().map(|o| self.obj_radius(&o.obj_type)).collect();
        let n = self.objects.len();
        for i in 0..n {
            for j in (i+1)..n {
                let r1 = col_radii[i];
                let r2 = col_radii[j];
                let d = self.objects[i].pos - self.objects[j].pos;
                let dist = d.len();
                let min_d = r1 + r2;
                if dist < min_d && dist > 1e-6 {
                    let n = d * (1.0 / dist);
                    let overlap = min_d - dist;
                    let m1 = self.objects[i].mass; let m2 = self.objects[j].mass;
                    let mt = m1 + m2;
                    self.objects[i].pos = self.objects[i].pos + n * (overlap * m2 / mt);
                    self.objects[j].pos = self.objects[j].pos - n * (overlap * m1 / mt);
                    let vn1 = self.objects[i].vel.dot(&n);
                    let vn2 = self.objects[j].vel.dot(&n);
                    let vn1n = (vn1*(m1-m2) + 2.0*m2*vn2)/mt;
                    let vn2n = (vn2*(m2-m1) + 2.0*m1*vn1)/mt;
                    self.objects[i].vel = self.objects[i].vel + n * ((vn1n - vn1) * self.bounce_coef);
                    self.objects[j].vel = self.objects[j].vel + n * ((vn2n - vn2) * self.bounce_coef);
                }
            }
        }
        self.step_count += 1;
    }

    // ── 3D Rendering: simple orthographic projection + flat shading ──

    fn project(&self, p: &Vec3) -> (f32, f32, f32) {
        // Camera basis vectors
        let forward = (self.cam_target - self.cam_pos).norm();
        let right = forward.cross(&self.cam_up).norm();
        let up = right.cross(&forward).norm();
        let rel = *p - self.cam_pos;
        let px = rel.dot(&right);
        let py = rel.dot(&up);
        let pz = rel.dot(&forward);
        (px, py, pz)
    }

    fn render(&self) -> image::RgbImage {
        let s = self.image_size as u32;
        let half = s as f32 * 0.5;
        let scale = s as f32 / 80.0; // world 80 units → screen
        let mut img = image::RgbImage::new(s, s);

        // Background gradient (sky-like)
        for y in 0..s {
            for x in 0..s {
                let t = y as f32 / s as f32;
                let r = (60.0 + t * 30.0) as u8;
                let g = (70.0 + t * 30.0) as u8;
                let b = (90.0 + t * 40.0) as u8;
                img.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }

        // Ground plane (simple grid lines at y=0)
        let grid_color = [60u8, 65, 70];
        for i in (0..=100).step_by(10) {
            let (gx, gy, gz) = self.project(&Vec3(i as f32, 0.0, 0.0));
            let sx = (half + gx * scale) as i32;
            let sy = (half - gy * scale) as i32;
            if sx >= 0 && sx < s as i32 && sy >= 0 && sy < s as i32 {
                img.put_pixel(sx as u32, sy as u32, image::Rgb(grid_color));
            }
            let (gx2, gy2, _) = self.project(&Vec3(i as f32, 0.0, 100.0));
            let sx2 = (half + gx2 * scale) as i32;
            let sy2 = (half - gy2 * scale) as i32;
            if sx2 >= 0 && sx2 < s as i32 && sy2 >= 0 && sy2 < s as i32 {
                img.put_pixel(sx2 as u32, sy2 as u32, image::Rgb(grid_color));
            }
            let (gz1, gz1y, _) = self.project(&Vec3(0.0, 0.0, i as f32));
            let (gz2, gz2y, _) = self.project(&Vec3(100.0, 0.0, i as f32));
            if (gz1 as i32) >= 0 && (gz1 as i32) < s as i32 && (gz1y as i32) >= 0 && (gz1y as i32) < s as i32 {
                img.put_pixel(gz1 as u32, (half - gz1y * scale) as u32, image::Rgb(grid_color));
            }
        }

        // Sort objects by depth (far to near)
        let mut indices: Vec<usize> = (0..self.objects.len()).collect();
        indices.sort_by(|&a, &b| {
            let (_, _, za) = self.project(&self.objects[a].pos);
            let (_, _, zb) = self.project(&self.objects[b].pos);
            zb.partial_cmp(&za).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Draw objects
        for &idx in &indices {
            let o = &self.objects[idx];
            let (px, py, pz) = self.project(&o.pos);
            let sx = (half + px * scale) as i32;
            let sy = (half - py * scale) as i32;
            let radius_px = match &o.obj_type {
                ObjType::Ball{radius} => (*radius * scale) as i32,
                ObjType::Cube{half_size} => (*half_size * scale * 1.2) as i32,
            }.max(2);

            // Lighting: simple directional light
            let light = (-self.light_dir.dot(&Vec3(0.0, 1.0, 0.0).norm())).max(0.3).min(1.0);

            for dy in -radius_px..=radius_px {
                for dx in -radius_px..=radius_px {
                    if dx*dx + dy*dy > radius_px*radius_px { continue; }
                    let ix = sx + dx;
                    let iy = sy + dy;
                    if ix < 0 || iy < 0 || ix >= s as i32 || iy >= s as i32 { continue; }
                    // Edge darkening for 3D appearance
                    let dist = ((dx*dx + dy*dy) as f32).sqrt() / radius_px as f32;
                    let shade = light * (1.0 - dist * 0.4);
                    img.put_pixel(ix as u32, iy as u32, image::Rgb([
                        (o.color[0] as f32 * shade).min(255.0) as u8,
                        (o.color[1] as f32 * shade).min(255.0) as u8,
                        (o.color[2] as f32 * shade).min(255.0) as u8,
                    ]));
                }
            }

            // Draw shadow on ground
            if o.pos.1 > 1.0 {
                let shadow_pos = Vec3(o.pos.0, 0.5, o.pos.2);
                let (spx, spy, _) = self.project(&shadow_pos);
                let ssx = (half + spx * scale) as i32;
                let ssy = (half - spy * scale) as i32;
                let sr = (radius_px as f32 * 0.6) as i32;
                for dy in -sr..=sr {
                    for dx in -sr..=sr {
                        if dx*dx+dy*dy > sr*sr { continue; }
                        let ix = ssx + dx; let iy = ssy + dy;
                        if ix < 0 || iy < 0 || ix >= s as i32 || iy >= s as i32 { continue; }
                        let existing = img.get_pixel(ix as u32, iy as u32);
                        let dark = ((existing[0] as f32) * 0.85) as u8;
                        img.put_pixel(ix as u32, iy as u32, image::Rgb([dark, dark, existing[2]]));
                    }
                }
            }
        }
        img
    }

    fn to_tensor<B: Backend>(&self) -> Tensor<B, 3> {
        let img = self.render();
        let s = self.image_size;
        let mut d = Vec::with_capacity(3 * s * s);
        for c in 0..3 {
            for y in 0..s { for x in 0..s {
                d.push(img.get_pixel(x as u32, y as u32)[c] as f32 / 255.0);
            }}
        }
        Tensor::<B,1>::from_floats(d.as_slice(), &Default::default()).reshape([3, s, s])
    }
}

impl Environment for Object3DWorld {
    fn reset<B: Backend>(&mut self, _device: &B::Device) -> Tensor<B, 3> {
        self.step_count = 0;
        self.init_scene(self.objects.len());
        self.to_tensor()
    }

    fn step<B: Backend>(&mut self, action: &[f32], _device: &B::Device) -> (Tensor<B, 3>, f32, bool) {
        let act = if action.is_empty() { 0.0 } else { action[0].clamp(-1.0, 1.0) };
        self.physics_step(act);
        let reward = self.compute_reward();
        (self.to_tensor(), reward, self.step_count >= self.max_steps)
    }

    fn obs_shape(&self) -> [usize; 3] { [self.image_channels, self.image_size, self.image_size] }
    fn action_dim(&self) -> usize { 1 }
    fn max_steps(&self) -> usize { self.max_steps }
}

impl Object3DWorld {
    fn compute_reward(&self) -> f32 {
        if self.objects.is_empty() { return 0.0; }
        // Reward: height of player object + avoid boundaries
        let p = &self.objects[0];
        let height_bonus = (p.pos.1 / 50.0).min(1.0);
        let edge_penalty = if p.pos.0 < 10.0 || p.pos.0 > 90.0 || p.pos.2 < 10.0 || p.pos.2 > 90.0 { -0.1 } else { 0.0 };
        height_bonus + edge_penalty
    }
}
