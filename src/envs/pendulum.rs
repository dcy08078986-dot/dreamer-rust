#![allow(dead_code, unused_variables)]
use crate::envs::Environment;
use burn::tensor::{backend::Backend, Tensor};
use rand::Rng;
use rand::SeedableRng;

pub struct PendulumEnv {
    theta: f32,
    theta_dot: f32,
    step_count: usize,
    max_steps: usize,
    action_dim: usize,
    obs_shape: [usize; 3],
    render_size: u32,
    rng: rand::rngs::StdRng,
}

fn angle_normalize(x: f32) -> f32 {
    let mut y = x;
    while y > std::f32::consts::PI { y -= 2.0 * std::f32::consts::PI; }
    while y < -std::f32::consts::PI { y += 2.0 * std::f32::consts::PI; }
    y
}

/// Bresenham line: draw line pixels into a buffer.
fn draw_line(buf: &mut [u8], w: u32, _h: u32, x0: i32, y0: i32, x1: i32, y1: i32, half_w: i32, color: (u8, u8, u8)) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    loop {
        // draw a "thick" pixel: fill a square of (2*half_w+1)×(2*half_w+1)
        for dy2 in -half_w..=half_w {
            for dx2 in -half_w..=half_w {
                let px = (x + dx2) as u32;
                let py = (y + dy2) as u32;
                if px < w && py < w {
                    let idx = ((py * w + px) * 3) as usize;
                    buf[idx] = color.0;
                    buf[idx + 1] = color.1;
                    buf[idx + 2] = color.2;
                }
            }
        }
        if x == x1 && y == y1 { break; }
        let e2 = 2 * err;
        if e2 >= dy { err += dy; x += sx; }
        if e2 <= dx { err += dx; y += sy; }
    }
}

/// Midpoint circle algorithm: draw filled circle.
fn draw_circle(buf: &mut [u8], w: u32, _h: u32, cx: i32, cy: i32, r: i32, color: (u8, u8, u8)) {
    let r2 = r * r;
    for dy in -r..=r {
        let dx_max = ((r2 - dy * dy) as f64).sqrt() as i32;
        for dx in -dx_max..=dx_max {
            let px = (cx + dx) as u32;
            let py = (cy + dy) as u32;
            if px < w && py < w {
                let idx = ((py * w + px) * 3) as usize;
                buf[idx] = color.0;
                buf[idx + 1] = color.1;
                buf[idx + 2] = color.2;
            }
        }
    }
}

impl PendulumEnv {
    pub fn new(
        max_steps: usize,
        action_dim: usize,
        image_channels: usize,
        image_size: usize,
        seed: u64,
    ) -> Self {
        Self {
            theta: 0.0, theta_dot: 0.0,
            step_count: 0,
            max_steps,
            action_dim,
            obs_shape: [image_channels, image_size, image_size],
            render_size: image_size as u32,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    fn render<B: Backend>(&self, device: &B::Device) -> Tensor<B, 3> {
        let s = self.render_size as i32;
        let s_u = self.render_size as usize;

        // Allocate flat RGB buffer, fill with black
        let mut buf = vec![0u8; 3 * s_u * s_u];

        let pivot_x = s as f32 / 2.0;
        let pivot_y = s as f32 * 0.25;
        let rod_len = s as f32 * 0.3;

        let bob_x = pivot_x + rod_len * self.theta.sin();
        let bob_y = pivot_y - rod_len * self.theta.cos();

        let px = pivot_x as i32;
        let py = pivot_y as i32;
        let bx = bob_x as i32;
        let by = bob_y as i32;

        // Draw rod (white, width 1 → half_w=1 → 3px thick)
        draw_line(&mut buf, s as u32, s as u32, px, py, bx, by, 1, (255, 255, 255));
        // Draw bob (red, radius 4)
        draw_circle(&mut buf, s as u32, s as u32, bx, by, 4, (255, 100, 100));
        // Draw pivot (gray, radius 2)
        draw_circle(&mut buf, s as u32, s as u32, px, py, 2, (200, 200, 200));

        // Convert RGB buffer → NCHW f32 tensor [0, 1]
        let total = 3 * s_u * s_u;
        let mut data = vec![0.0f32; total];
        for c in 0..3usize {
            for y in 0..s_u {
                for x in 0..s_u {
                    let src_idx = (y * s_u + x) * 3 + c;
                    data[c * s_u * s_u + y * s_u + x] = buf[src_idx] as f32 / 255.0;
                }
            }
        }
        Tensor::<B, 1>::from_floats(data.as_slice(), device)
            .reshape([3, s_u, s_u])
    }
}

impl Environment for PendulumEnv {
    fn reset<B: Backend>(&mut self, device: &B::Device) -> Tensor<B, 3> {
        self.theta = self.rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI);
        self.theta_dot = self.rng.gen_range(-1.0..1.0);
        self.step_count = 0;
        self.render::<B>(device)
    }

    fn step<B: Backend>(
        &mut self, action: &[f32], device: &B::Device,
    ) -> (Tensor<B, 3>, f32, bool) {
        let max_speed = 8.0;
        let max_torque = 2.0;
        let dt = 0.05;
        let g = 10.0;
        let m = 1.0;
        let l = 1.0;

        let u = action[0].clamp(-1.0, 1.0) * max_torque;
        let cost = angle_normalize(self.theta).powi(2)
            + 0.1 * self.theta_dot.powi(2) + 0.001 * u.powi(2);

        let new_theta_dot = self.theta_dot
            + (3.0 * g / (2.0 * l) * self.theta.sin()
                + 3.0 / (m * l.powi(2)) * u) * dt;

        self.theta_dot = new_theta_dot.clamp(-max_speed, max_speed);
        self.theta += self.theta_dot * dt;
        self.step_count += 1;

        let done = self.step_count >= self.max_steps;
        let obs = self.render::<B>(device);
        (obs, -cost, done)
    }

    fn obs_shape(&self) -> [usize; 3] { self.obs_shape }
    fn action_dim(&self) -> usize { self.action_dim }
    fn max_steps(&self) -> usize { self.max_steps }
}
