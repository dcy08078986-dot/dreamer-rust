#![allow(dead_code, unused_variables)]
use crate::envs::Environment;
use burn::tensor::{backend::Backend, Tensor};
use rand::Rng;

pub struct VisualNavigation {
    // agent state (physical coords [0, 10])
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,

    // target position
    target_x: f32,
    target_y: f32,

    // obstacles: list of (cx, cy, w, h) in physical coords
    obstacles: Vec<(f32, f32, f32, f32)>,

    // episode step counter
    step_count: usize,

    // config
    max_steps: usize,
    action_dim: usize,
    obs_shape: [usize; 3],
    render_size: u32,

    // rng
    rng: rand::rngs::StdRng,
}

impl VisualNavigation {
    pub fn new(
        max_steps: usize,
        action_dim: usize,
        image_channels: usize,
        image_size: usize,
        num_obstacles: usize,
        seed: u64,
    ) -> Self {
        use rand::SeedableRng;
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        let obstacles = Self::generate_obstacles(num_obstacles);

        Self {
            x: 5.0,
            y: 5.0,
            vx: 0.0,
            vy: 0.0,
            target_x: 5.0,
            target_y: 5.0,
            obstacles,
            step_count: 0,
            max_steps,
            action_dim,
            obs_shape: [image_channels, image_size, image_size],
            render_size: image_size as u32,
            rng,
        }
    }

    fn generate_obstacles(n: usize) -> Vec<(f32, f32, f32, f32)> {
        // Fixed placement to keep determinism
        let candidates = vec![
            (2.0, 2.0, 1.5, 0.5),
            (7.0, 3.0, 0.5, 2.0),
            (4.0, 7.0, 2.0, 0.5),
            (8.0, 8.0, 1.0, 1.0),
            (1.0, 6.0, 0.5, 1.5),
        ];
        candidates.into_iter().take(n).collect()
    }

    fn is_inside_obstacle(&self, px: f32, py: f32) -> bool {
        for &(cx, cy, w, h) in &self.obstacles {
            let half_w = w / 2.0;
            let half_h = h / 2.0;
            if px >= cx - half_w && px <= cx + half_w
                && py >= cy - half_h && py <= cy + half_h
            {
                return true;
            }
        }
        false
    }

    fn random_target(&mut self) {
        loop {
            let tx = self.rng.gen_range(0.5..9.5);
            let ty = self.rng.gen_range(0.5..9.5);
            let dist_to_agent = ((tx - self.x).powi(2) + (ty - self.y).powi(2)).sqrt();
            if dist_to_agent > 3.0 && !self.is_inside_obstacle(tx, ty) {
                self.target_x = tx;
                self.target_y = ty;
                break;
            }
        }
    }

    fn physics_step(&mut self, fx: f32, fy: f32) {
        let force_scale = 3.0;
        let friction = 1.5;
        let dt = 0.1;

        let ax = fx * force_scale - friction * self.vx;
        let ay = fy * force_scale - friction * self.vy;

        self.vx += ax * dt;
        self.vy += ay * dt;
        self.x += self.vx * dt;
        self.y += self.vy * dt;

        // clamp to bounds, zero velocity on collision
        if self.x < 0.0 {
            self.x = 0.0;
            self.vx = 0.0;
        }
        if self.x > 10.0 {
            self.x = 10.0;
            self.vx = 0.0;
        }
        if self.y < 0.0 {
            self.y = 0.0;
            self.vy = 0.0;
        }
        if self.y > 10.0 {
            self.y = 10.0;
            self.vy = 0.0;
        }
    }

    fn render<B: Backend>(&self, device: &B::Device) -> Tensor<B, 3> {
        let s = self.render_size;
        let mut img = image::RgbImage::new(s, s);

        let scale = s as f32 / 10.0; // pixels per physical unit

        for py in 0..s {
            for px in 0..s {
                let wx = px as f32 / scale;
                let wy = py as f32 / scale;

                // check agent
                let dist_agent = ((wx - self.x).powi(2) + (wy - self.y).powi(2)).sqrt();
                // check target
                let dist_target =
                    ((wx - self.target_x).powi(2) + (wy - self.target_y).powi(2)).sqrt();
                // check obstacles
                let in_obstacle = self.is_inside_obstacle(wx, wy);

                let (r, g, b) = if dist_agent < 0.25 {
                    (255u8, 255u8, 255u8) // white agent
                } else if dist_target < 0.25 {
                    (0u8, 255u8, 0u8) // green target
                } else if in_obstacle {
                    (128u8, 128u8, 128u8) // gray obstacle
                } else {
                    (0u8, 0u8, 0u8) // black background
                };

                img.put_pixel(px, py, image::Rgb([r, g, b]));
            }
        }

        // Convert RgbImage → flat f32 in NCHW order
        let mut data = vec![0.0f32; (3 * s * s) as usize];
        let s_usize = s as usize;
        for c in 0..3usize {
            for y in 0..s_usize {
                for x in 0..s_usize {
                    let pixel = img.get_pixel(x as u32, y as u32);
                    let val = pixel[c] as f32 / 255.0;
                    data[c * s_usize * s_usize + y * s_usize + x] = val;
                }
            }
        }

        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([3, s as usize, s as usize])
    }
}

impl Environment for VisualNavigation {
    fn reset<B: Backend>(&mut self, device: &B::Device) -> Tensor<B, 3> {
        self.x = 5.0;
        self.y = 5.0;
        self.vx = 0.0;
        self.vy = 0.0;
        self.step_count = 0;
        self.random_target();
        self.render::<B>(device)
    }

    fn step<B: Backend>(
        &mut self,
        action: &[f32],
        device: &B::Device,
    ) -> (Tensor<B, 3>, f32, bool) {
        let fx = action[0].clamp(-1.0, 1.0);
        let fy = action[1].clamp(-1.0, 1.0);

        self.physics_step(fx, fy);
        self.step_count += 1;

        let dist = ((self.x - self.target_x).powi(2) + (self.y - self.target_y).powi(2)).sqrt();
        let done = dist < 0.5 || self.step_count >= self.max_steps;
        // dense distance reward + sparse success bonus
        let reward = -dist * 0.03 + if dist < 0.5 { 1.0 } else { 0.0 };

        let obs = self.render::<B>(device);
        (obs, reward, done)
    }

    fn obs_shape(&self) -> [usize; 3] {
        self.obs_shape
    }

    fn action_dim(&self) -> usize {
        self.action_dim
    }

    fn max_steps(&self) -> usize {
        self.max_steps
    }
}
