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
        let mut pixels = vec![0.0f32; self.image_channels * size * size];

        // 背景渐变（天空到地面）
        for y in 0..size {
            let sky_intensity = (size - y) as f32 / size as f32;
            for x in 0..size {
                let idx = (y * size + x) * self.image_channels;
                // 天空蓝色渐变
                pixels[idx] = sky_intensity * 0.3;     // R
                pixels[idx + 1] = sky_intensity * 0.6; // G
                pixels[idx + 2] = sky_intensity * 0.9; // B
            }
        }

        // 绘制地面线
        let ground_y = size - 1;
        for x in 0..size {
            let idx = (ground_y * size + x) * self.image_channels;
            pixels[idx] = 0.2;
            pixels[idx + 1] = 0.8;
            pixels[idx + 2] = 0.2;
        }

        // 绘制球（红色）
        let ball_x = (self.x * size as f32) as usize;
        let ball_y = ((1.0 - self.y) * size as f32) as usize; // 翻转Y轴
        let ball_radius = (size / 16).max(2);

        for dy in -(ball_radius as i32)..=(ball_radius as i32) {
            for dx in -(ball_radius as i32)..=(ball_radius as i32) {
                if dx * dx + dy * dy <= (ball_radius * ball_radius) as i32 {
                    let px = (ball_x as i32 + dx).clamp(0, size as i32 - 1) as usize;
                    let py = (ball_y as i32 + dy).clamp(0, size as i32 - 1) as usize;
                    let idx = (py * size + px) * self.image_channels;

                    // 红色球，带一点高光
                    let dist_ratio = ((dx * dx + dy * dy) as f32).sqrt() / ball_radius as f32;
                    let highlight = (1.0 - dist_ratio).max(0.0);

                    pixels[idx] = 0.9 + highlight * 0.1;     // R
                    pixels[idx + 1] = 0.1 + highlight * 0.5; // G
                    pixels[idx + 2] = 0.1 + highlight * 0.3; // B
                }
            }
        }

        // 绘制速度向量（箭头）
        if self.vx.abs() > 0.01 || self.vy.abs() > 0.01 {
            let arrow_len = 10.min(size / 8);
            let end_x = ball_x as i32 + (self.vx * arrow_len as f32 * 50.0) as i32;
            let end_y = ball_y as i32 - (self.vy * arrow_len as f32 * 50.0) as i32;

            // 简单的线条绘制
            self.draw_line(&mut pixels, ball_x, ball_y,
                          end_x.clamp(0, size as i32 - 1) as usize,
                          end_y.clamp(0, size as i32 - 1) as usize,
                          size, [1.0, 1.0, 0.0]); // 黄色箭头
        }

        pixels
    }

    fn draw_line(&self, pixels: &mut [f32], x0: usize, y0: usize, x1: usize, y1: usize, size: usize, color: [f32; 3]) {
        let dx = (x1 as i32 - x0 as i32).abs();
        let dy = (y1 as i32 - y0 as i32).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx - dy;

        let mut x = x0 as i32;
        let mut y = y0 as i32;

        for _ in 0..100 { // 限制迭代次数
            if x >= 0 && x < size as i32 && y >= 0 && y < size as i32 {
                let idx = (y as usize * size + x as usize) * self.image_channels;
                pixels[idx] = color[0];
                pixels[idx + 1] = color[1];
                pixels[idx + 2] = color[2];
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

        let pixels = self.render();
        let [c, h, w] = self.obs_shape();
        Tensor::<B, 1>::from_floats(pixels.as_slice(), device).reshape([c, h, w])
    }

    fn step<B: Backend>(&mut self, action: &[f32], device: &B::Device) -> (Tensor<B, 3>, f32, bool) {
        let force_x = action[0].clamp(-1.0, 1.0);

        self.physics_step(force_x);
        self.step_count += 1;

        let reward = self.compute_reward();
        let done = self.step_count >= self.max_steps || self.y <= 0.01;

        let pixels = self.render();
        let [c, h, w] = self.obs_shape();
        let obs = Tensor::<B, 1>::from_floats(pixels.as_slice(), device).reshape([c, h, w]);

        (obs, reward, done)
    }
}
