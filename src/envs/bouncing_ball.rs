#![allow(dead_code, unused_variables)]
use crate::envs::Environment;
use burn::tensor::{backend::Backend, Tensor};
use rand::Rng;

/// 弹球环境 - 专为测试世界模型设计
///
/// 特点：
/// 1. 简单的物理规律：重力 + 弹性碰撞
/// 2. 可预测的动态：世界模型容易学习
/// 3. 视觉清晰：球的位置和速度直观可见
/// 4. 连续控制：左右施加力
/// 5. 密集奖励：球越高奖励越大
pub struct BouncingBall {
    // 球的状态
    x: f32,      // 水平位置 [0, 1]
    y: f32,      // 垂直位置 [0, 1]
    vx: f32,     // 水平速度
    vy: f32,     // 垂直速度

    // 环境参数
    gravity: f32,
    bounce_damping: f32,
    control_force: f32,

    // Episode管理
    step_count: usize,
    max_steps: usize,

    // 渲染参数
    image_size: usize,
    image_channels: usize,

    // ── P1: trail + background variation ──
    trail: Vec<(f32, f32)>,   // 最近 N 帧球的位置 (x, y)
    trail_len: usize,
    cloud_seed: u64,           // per-episode seed for cloud placement

    rng: rand::rngs::StdRng,
}

impl BouncingBall {
    pub fn new(
        max_steps: usize,
        action_dim: usize,
        image_channels: usize,
        image_size: usize,
        seed: u64,
    ) -> Self {
        use rand::SeedableRng;
        let rng = rand::rngs::StdRng::seed_from_u64(seed);

        Self {
            x: 0.5,
            y: 0.5,
            vx: 0.0,
            vy: 0.0,
            gravity: 0.002,
            bounce_damping: 0.8,
            control_force: 0.001,
            step_count: 0,
            max_steps,
            image_size,
            image_channels,
            trail: Vec::with_capacity(16),
            trail_len: 12,
            cloud_seed: seed,
            rng,
        }
    }

    fn physics_step(&mut self, force_x: f32) {
        // 施加控制力
        self.vx += force_x * self.control_force;

        // 施加重力
        self.vy -= self.gravity;

        // 更新位置
        self.x += self.vx;
        self.y += self.vy;

        // 左右边界碰撞
        if self.x < 0.0 {
            self.x = 0.0;
            self.vx = -self.vx * self.bounce_damping;
        } else if self.x > 1.0 {
            self.x = 1.0;
            self.vx = -self.vx * self.bounce_damping;
        }

        // 上下边界碰撞
        if self.y < 0.0 {
            self.y = 0.0;
            self.vy = -self.vy * self.bounce_damping;
        } else if self.y > 1.0 {
            self.y = 1.0;
            self.vy = -self.vy * self.bounce_damping;
        }

        // 速度衰减（空气阻力）
        self.vx *= 0.99;
        self.vy *= 0.99;
    }

    fn compute_reward(&self) -> f32 {
        // 奖励 = 高度 + 保持在中央的奖励
        let height_reward = self.y * 10.0;
        let center_reward = 1.0 - (self.x - 0.5).abs() * 2.0;
        height_reward + center_reward * 0.5
    }

    fn render(&self) -> Vec<f32> {
        let size = self.image_size;
        let n = size * size;
        let mut pixels = vec![0.0f32; self.image_channels * n];

        // ── static sky gradient ──
        for y in 0..size {
            let sky = (size - y) as f32 / size as f32;
            for x in 0..size {
                let dst = y * size + x;
                pixels[dst] = sky * 0.3;
                pixels[n + dst] = sky * 0.6;
                pixels[2 * n + dst] = sky * 0.9;
            }
        }

        // ── static ground line ──
        let ground_y = size - 1;
        for x in 0..size {
            let dst = ground_y * size + x;
            pixels[dst] = 0.2;
            pixels[n + dst] = 0.8;
            pixels[2 * n + dst] = 0.2;
        }

        // ── P1: ball trail (fading circles) ──
        let trail_len = self.trail.len();
        for (i, &(tx, ty)) in self.trail.iter().enumerate() {
            let alpha = 0.15 * (i + 1) as f32 / trail_len as f32; // newer = brighter
            let trail_radius = (size / 24).max(2);
            let trail_col = [0.9 * alpha, 0.2 * alpha, 0.15 * alpha];
            self.draw_ball_chw(&mut pixels, tx, ty, trail_radius, size, n, trail_col);
        }

        // ── P0: ball (larger radius = size/8) ──
        let ball_radius = (size / 8).max(4);
        self.draw_ball_chw(&mut pixels, self.x, self.y, ball_radius, size, n,
            [0.9, 0.15, 0.1]);

        // ── velocity arrow ──
        if self.vx.abs() > 0.005 || self.vy.abs() > 0.005 {
            let bx = (self.x * size as f32) as usize;
            let by = ((1.0 - self.y) * size as f32) as usize;
            let al = 12.min(size / 6);
            let ex = bx as i32 + (self.vx * al as f32 * 40.0) as i32;
            let ey = by as i32 - (self.vy * al as f32 * 40.0) as i32;
            self.draw_line(&mut pixels, bx, by,
                ex.clamp(0, size as i32 - 1) as usize,
                ey.clamp(0, size as i32 - 1) as usize,
                size, n, [1.0, 1.0, 0.0]);
        }

        pixels
    }

    /// Draw a filled circle (CHW) at normalized coords (x,y ∈ [0,1])
    fn draw_ball_chw(&self, p: &mut [f32], cx: f32, cy: f32, radius: usize,
                     size: usize, n: usize, color: [f32; 3]) {
        let px = (cx * size as f32) as usize;
        let py = ((1.0 - cy) * size as f32) as usize;
        let r = radius as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    let sx = (px as i32 + dx).clamp(0, size as i32 - 1) as usize;
                    let sy = (py as i32 + dy).clamp(0, size as i32 - 1) as usize;
                    let dst = sy * size + sx;
                    let dr = ((dx * dx + dy * dy) as f32).sqrt() / radius as f32;
                    let hl = (1.0 - dr).max(0.0);
                    // blend: add with highlight, saturate at 1.0
                    p[dst] = (p[dst] + color[0] + hl * 0.2).min(1.0);
                    p[n + dst] = (p[n + dst] + color[1] + hl * 0.3).min(1.0);
                    p[2 * n + dst] = (p[2 * n + dst] + color[2] + hl * 0.2).min(1.0);
                }
            }
        }
    }

    /// Draw random oval clouds (deterministic per episode via cloud_seed + step_count)
    fn draw_clouds(&self, p: &mut [f32], size: usize, n: usize) {
        // Simple pseudo-random based on cloud_seed
        let mut h: u64 = self.cloud_seed.wrapping_add(self.step_count as u64);
        for _ in 0..3 {
            h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let cx = ((h >> 32) as usize) % size;
            h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let cy = ((h >> 32) as usize) % (size / 3); // clouds in upper third
            let rw = 4 + ((h >> 16) as usize % 10); // width
            let rh: usize = 2 + ((h >> 40) as usize % 4); // height
            for dy in -(rh as i32)..=(rh as i32) {
                for dx in -(rw as i32)..=(rw as i32) {
                    let d = (dx * dx) as f32 / (rw * rw) as f32
                          + (dy * dy) as f32 / (rh * rh) as f32;
                    if d <= 1.0 {
                        let sx = (cx as i32 + dx).clamp(0, size as i32 - 1) as usize;
                        let sy = (cy as i32 + dy).clamp(0, size as i32 - 1) as usize;
                        let dst = sy * size + sx;
                        let alpha = 0.08 * (1.0 - d);
                        p[dst] = (p[dst] + alpha).min(1.0);
                        p[n + dst] = (p[n + dst] + alpha).min(1.0);
                        p[2 * n + dst] = (p[2 * n + dst] + alpha).min(1.0);
                    }
                }
            }
        }
    }

    fn push_trail(&mut self) {
        self.trail.push((self.x, self.y));
        if self.trail.len() > self.trail_len {
            self.trail.remove(0);
        }
    }

    /// True physical state (x, y, vx, vy) — diagnostics only.
    pub fn state(&self) -> [f32; 4] {
        [self.x, self.y, self.vx, self.vy]
    }

    fn draw_line(&self, pixels: &mut [f32], x0: usize, y0: usize, x1: usize, y1: usize, size: usize, n: usize, color: [f32; 3]) {
        let dx = (x1 as i32 - x0 as i32).abs();
        let dy = (y1 as i32 - y0 as i32).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx - dy;

        let mut x = x0 as i32;
        let mut y = y0 as i32;

        for _ in 0..100 { // limit iterations
            if x >= 0 && x < size as i32 && y >= 0 && y < size as i32 {
                let dst = (y as usize) * size + (x as usize);
                pixels[dst] = color[0];
                pixels[n + dst] = color[1];
                pixels[2 * n + dst] = color[2];
            }

            if x == x1 as i32 && y == y1 as i32 { break; }

            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                y += sy;
            }
        }
    }
}

impl Environment for BouncingBall {
    fn obs_shape(&self) -> [usize; 3] {
        [self.image_channels, self.image_size, self.image_size]
    }

    fn action_dim(&self) -> usize {
        1 // 只有水平控制力
    }

    fn max_steps(&self) -> usize {
        self.max_steps
    }

    fn reset<B: Backend>(&mut self, device: &B::Device) -> Tensor<B, 3> {
        // 随机初始化球的位置
        self.x = self.rng.gen_range(0.3..0.7);
        self.y = self.rng.gen_range(0.5..0.8);
        self.vx = self.rng.gen_range(-0.01..0.01);
        self.vy = 0.0;
        self.step_count = 0;
        self.trail.clear();
        self.cloud_seed = self.rng.r#gen(); // new cloud positions per episode

        let pixels = self.render();
        let [c, h, w] = self.obs_shape();
        Tensor::<B, 1>::from_floats(pixels.as_slice(), device).reshape([c, h, w])
    }

    fn step<B: Backend>(&mut self, action: &[f32], device: &B::Device) -> (Tensor<B, 3>, f32, bool) {
        let force_x = action[0].clamp(-1.0, 1.0);

        self.physics_step(force_x);
        self.push_trail();
        self.step_count += 1;

        let reward = self.compute_reward();
        let on_ground = self.y <= 0.01 && self.vy.abs() < 0.003;
        let done = self.step_count >= self.max_steps || on_ground;

        let pixels = self.render();
        let [c, h, w] = self.obs_shape();
        let obs = Tensor::<B, 1>::from_floats(pixels.as_slice(), device).reshape([c, h, w]);

        (obs, reward, done)
    }
}
