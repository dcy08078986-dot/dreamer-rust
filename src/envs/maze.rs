use crate::envs::Environment;
use burn::tensor::{backend::Backend, Tensor};
use rand::Rng;
use rand::SeedableRng;

/// A partially-observable 2D grid maze. Agent sees a local window as a 64×64 image.
pub struct MazeEnv {
    grid: Vec<Vec<bool>>,    // true = wall, false = path
    width: usize, height: usize,
    ax: usize, ay: usize,    // agent position
    adir: usize,             // agent dir: 0=right, 1=down, 2=left, 3=up
    gx: usize, gy: usize,    // goal
    step_count: usize, max_steps: usize,
    image_size: usize,
    rng: rand::rngs::StdRng,
}

impl MazeEnv {
    pub fn new(width: usize, height: usize, max_steps: usize, image_size: usize, seed: u64) -> Self {
        let mut env = Self {
            grid: vec![vec![false; width]; height],
            width, height, ax: 0, ay: 0, adir: 0, gx: 0, gy: 0,
            step_count: 0, max_steps, image_size,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        };
        env.gen_maze();
        env.place_agent_goal();
        env
    }

    fn gen_maze(&mut self) {
        // Randomized DFS maze
        let w = self.width;
        let h = self.height;
        for y in 0..h { for x in 0..w { self.grid[y][x] = true; } } // all walls

        let mut stack = vec![(1, 1)];
        self.grid[1][1] = false;

        while let Some(&(cx, cy)) = stack.last() {
            let dirs = [(2, 0, 0), (0, 2, 1), (-2, 0, 2), (0, -2, 3)];
            let mut neighbors: Vec<_> = dirs.iter().filter_map(|&(dx, dy, _)| {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if nx > 0 && nx < w as i32 - 1 && ny > 0 && ny < h as i32 - 1
                    && self.grid[ny as usize][nx as usize] {
                    Some((nx as usize, ny as usize, dx as i32, dy as i32))
                } else { None }
            }).collect();

            if neighbors.is_empty() { stack.pop(); }
            else {
                let idx = self.rng.gen_range(0..neighbors.len());
                let (nx, ny, dx, dy) = neighbors[idx];
                self.grid[ny][nx] = false;
                self.grid[(ny as i32 - dy/2) as usize][(nx as i32 - dx/2) as usize] = false;
                stack.push((nx, ny));
            }
        }
    }

    fn place_agent_goal(&mut self) {
        loop {
            self.ax = self.rng.gen_range(1..self.width - 1);
            self.ay = self.rng.gen_range(1..self.height - 1);
            if !self.grid[self.ay][self.ax] { break; }
        }
        loop {
            self.gx = self.rng.gen_range(1..self.width - 1);
            self.gy = self.rng.gen_range(1..self.height - 1);
            let dist = (self.gx as i32 - self.ax as i32).abs() + (self.gy as i32 - self.ay as i32).abs();
            if dist > self.width as i32 / 2 && !self.grid[self.gy][self.gx] { break; }
        }
        self.adir = self.rng.gen_range(0..4);
    }

    fn render<B: Backend>(&self, device: &B::Device) -> Tensor<B, 3> {
        let s = self.image_size;
        let view: i32 = 7;
        let cell = s as i32 / view;
        let mut buf = vec![0u8; 3 * s * s];

        let (dx_f, dx_r, dy_f, dy_r): (i32, i32, i32, i32) = match self.adir {
            0 => (1, 0, 0, -1), 1 => (0, 1, 1, 0), 2 => (-1, 0, 0, 1), 3 => (0, -1, -1, 0),
            _ => (1, 0, 0, -1),
        };

        for vr in 0..view {
            for vc in 0..view {
                let wx = self.ax as i32 + vc * dx_r + vr * dx_f;
                let wy = self.ay as i32 + vc * dy_r + vr * dy_f;
                let (r, g, b): (u8, u8, u8) = if wx < 0 || wy < 0
                    || wx >= self.width as i32 || wy >= self.height as i32 { (0,0,0) }
                else if wx == self.gx as i32 && wy == self.gy as i32 { (0,255,0) }
                else if self.grid[wy as usize][wx as usize] { (100,100,100) }
                else { (50,50,50) };

                let ox = (view - 1 - vc) * cell;
                let oy = (view - 1 - vr) * cell;
                for dy in 0..cell {
                    for dx in 0..cell {
                        let px = (ox + dx) as usize;
                        let py = (oy + dy) as usize;
                        if px < s && py < s {
                            let idx = (py * s + px) * 3;
                            buf[idx] = r; buf[idx+1] = g; buf[idx+2] = b;
                        }
                    }
                }
            }
        }

        // Agent diamond in center cell
        let cx = (view / 2 + 1) * cell + cell / 2;
        let cy = (view / 2 + 1) * cell + cell / 2;
        let hs = cell / 3;
        for dy in -hs..=hs {
            for dx in -hs..=hs {
                if dx.abs() + dy.abs() <= hs {
                    let px = (cx + dx) as usize;
                    let py = (cy + dy) as usize;
                    if px < s && py < s {
                        let idx = (py * s + px) * 3;
                        buf[idx] = 255; buf[idx+1] = 0; buf[idx+2] = 0;
                    }
                }
            }
        }

        let mut data = vec![0.0f32; 3 * s * s];
        for ch in 0..3 { for y in 0..s { for x in 0..s {
            data[ch * s * s + y * s + x] = buf[(y * s + x) * 3 + ch] as f32 / 255.0;
        }}}
        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([3, s, s])
    }
}

impl Environment for MazeEnv {
    fn reset<B: Backend>(&mut self, device: &B::Device) -> Tensor<B, 3> {
        self.step_count = 0;
        self.place_agent_goal();
        self.render::<B>(device)
    }

    fn step<B: Backend>(&mut self, action: &[f32], device: &B::Device) -> (Tensor<B, 3>, f32, bool) {
        let act = action[0] as i32;
        match act {
            0 => self.adir = (self.adir + 3) % 4, // left
            1 => self.adir = (self.adir + 1) % 4, // right
            2 => { // forward
                let (dx, dy): (i32, i32) = match self.adir { 0=>(1,0), 1=>(0,1), 2=>(-1,0), 3=>(0,-1), _=>(0,0) };
                let nx = (self.ax as i32 + dx).max(0) as usize;
                let ny = (self.ay as i32 + dy).max(0) as usize;
                if nx < self.width && ny < self.height && !self.grid[ny][nx] {
                    self.ax = nx; self.ay = ny;
                }
            }
            _ => {}
        }
        self.step_count += 1;
        let done = (self.ax == self.gx && self.ay == self.gy) || self.step_count >= self.max_steps;
        let reward = if self.ax == self.gx && self.ay == self.gy {
            1.0 - 0.9 * (self.step_count as f32 / self.max_steps as f32)
        } else { 0.0 };
        (self.render::<B>(device), reward, done)
    }

    fn obs_shape(&self) -> [usize; 3] { [3, self.image_size, self.image_size] }
    fn action_dim(&self) -> usize { 1 }
    fn max_steps(&self) -> usize { self.max_steps }
}
