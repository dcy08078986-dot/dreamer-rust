#![allow(dead_code, unused_variables)]
use crate::envs::Environment;
use burn::tensor::{backend::Backend, Tensor};

// ── Simple inline RNG (avoids rand 2024 edition gen keyword conflict) ──

struct XorShift(u64);

impl XorShift {
    fn new(seed: u64) -> Self { XorShift(seed.wrapping_add(0x9E3779B97F4A7C15)) }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn f32(&mut self) -> f32 { (self.next() as f32) / (u64::MAX as f32) }
    fn bool(&mut self, p: f64) -> bool { self.f32() < p as f32 }
}

// ── Object types ──

#[derive(Clone, Debug)]
pub struct Ball {
    pub x: f32, pub y: f32, pub vx: f32, pub vy: f32,
    pub radius: f32, pub mass: f32, pub color: [u8; 3],
}

#[derive(Clone, Debug)]
pub struct Wall { pub x: f32, pub y: f32, pub w: f32, pub h: f32 }

#[derive(Clone, Debug)]
pub struct Target { pub x: f32, pub y: f32, pub radius: f32 }

#[derive(Clone, Debug)]
pub struct ObjectWorldState {
    pub balls: Vec<Ball>, pub walls: Vec<Wall>, pub target: Target,
    pub step: usize, pub collision_events: Vec<(usize, usize, usize)>,
}

// ── Environment ──

pub struct ObjectWorld {
    pub balls: Vec<Ball>,
    pub walls: Vec<Wall>,
    pub target: Target,
    gravity: f32, damping: f32, bounce_coef: f32,
    step_count: usize, max_steps: usize,
    collision_events: Vec<(usize, usize, usize)>,
    image_size: usize, image_channels: usize,
    trail: Vec<Vec<(f32, f32)>>, trail_len: usize,
    player_idx: usize, control_force: f32,
    rng: XorShift,
}

impl ObjectWorld {
    pub fn new(
        max_steps: usize, action_dim: usize, image_channels: usize, image_size: usize,
        seed: u64, num_balls: usize, num_walls: usize,
    ) -> Self {
        let mut rng = XorShift::new(seed);
        let mut obj = Self {
            balls: Vec::new(), walls: Vec::new(),
            target: Target { x: 0.5, y: 0.9, radius: 0.04 },
            gravity: 0.001, damping: 0.995, bounce_coef: 0.85,
            step_count: 0, max_steps, collision_events: Vec::new(),
            image_size, image_channels, trail: Vec::new(), trail_len: 15,
            player_idx: 0, control_force: 0.015, rng,
        };
        obj.init_objects(num_balls, num_walls);
        obj
    }

    fn init_objects(&mut self, num_balls: usize, num_walls: usize) {
        let colors: [[u8; 3]; 6] = [
            [255,80,80],[80,180,255],[80,255,120],[255,220,60],[200,80,255],[255,140,40]
        ];
        for i in 0..num_balls {
            let radius = 0.04 + self.rng.f32() * 0.02;
            loop {
                let x = 0.15 + self.rng.f32() * 0.7;
                let y = 0.1 + self.rng.f32() * 0.4;
                if self.balls.iter().all(|b| {
                    let d = ((x-b.x).powi(2) + (y-b.y).powi(2)).sqrt();
                    d >= radius + b.radius + 0.02
                }) {
                    self.balls.push(Ball {
                        x, y,
                        vx: (self.rng.f32() - 0.5) * 0.01,
                        vy: (self.rng.f32() - 0.5) * 0.01,
                        radius, mass: radius * radius,
                        color: colors[i % colors.len()],
                    });
                    break;
                }
            }
        }
        if !self.balls.is_empty() { self.balls[0].color = [60, 220, 220]; }
        // Walls
        for _ in 0..num_walls {
            let (x, y, w, h) = if self.rng.bool(0.5) {
                (0.1+self.rng.f32()*0.6, 0.2+self.rng.f32()*0.6, 0.25, 0.012)
            } else {
                (0.2+self.rng.f32()*0.6, 0.15+self.rng.f32()*0.5, 0.012, 0.2)
            };
            self.walls.push(Wall { x, y, w, h });
        }
        // Target
        loop {
            let tx = 0.2 + self.rng.f32() * 0.6;
            let ty = 0.75 + self.rng.f32() * 0.15;
            if self.balls.iter().all(|b| {
                ((tx-b.x).powi(2)+(ty-b.y).powi(2)).sqrt() >= b.radius + self.target.radius + 0.02
            }) { self.target = Target { x: tx, y: ty, radius: 0.04 }; break; }
        }
        self.trail = vec![Vec::new(); self.balls.len()];
    }

    pub fn num_balls(&self) -> usize { self.balls.len() }
    pub fn state(&self) -> ObjectWorldState {
        ObjectWorldState {
            balls: self.balls.clone(), walls: self.walls.clone(),
            target: self.target.clone(), step: self.step_count,
            collision_events: self.collision_events.clone(),
        }
    }

    fn physics_step(&mut self, action: f32) {
        if !self.balls.is_empty() { self.balls[self.player_idx].vx += action * self.control_force; }
        for b in &mut self.balls {
            b.vy -= self.gravity;
            b.vx *= self.damping; b.vy *= self.damping;
            b.x += b.vx; b.y += b.vy;
        }
        for b in &mut self.balls {
            if b.x - b.radius < 0.0 { b.x = b.radius; b.vx = b.vx.abs() * self.bounce_coef; }
            if b.x + b.radius > 1.0 { b.x = 1.0 - b.radius; b.vx = -b.vx.abs() * self.bounce_coef; }
            if b.y - b.radius < 0.0 { b.y = b.radius; b.vy = b.vy.abs() * self.bounce_coef; }
            if b.y + b.radius > 1.0 { b.y = 1.0 - b.radius; b.vy = -b.vy.abs() * self.bounce_coef; }
        }
        // Wall bounce
        for b in &mut self.balls {
            for w in &self.walls {
                let hw = w.w * 0.5; let hh = w.h * 0.5;
                let cx = b.x.clamp(w.x - hw - b.radius, w.x + hw + b.radius);
                let cy = b.y.clamp(w.y - hh - b.radius, w.y + hh + b.radius);
                let dx = b.x - cx; let dy = b.y - cy;
                let dist = (dx*dx + dy*dy).sqrt();
                if dist < b.radius && dist > 1e-6 {
                    let nx = dx/dist; let ny = dy/dist;
                    b.x += nx * (b.radius - dist); b.y += ny * (b.radius - dist);
                    let vn = b.vx*nx + b.vy*ny;
                    if vn < 0.0 { b.vx -= 2.0*vn*nx*self.bounce_coef; b.vy -= 2.0*vn*ny*self.bounce_coef; }
                }
            }
        }
        // Ball-ball collision
        let n = self.balls.len();
        for i in 0..n { for j in i+1..n {
            let dx = self.balls[i].x - self.balls[j].x;
            let dy = self.balls[i].y - self.balls[j].y;
            let dist = (dx*dx+dy*dy).sqrt();
            let min_d = self.balls[i].radius + self.balls[j].radius;
            if dist < min_d && dist > 1e-6 {
                self.collision_events.push((self.step_count, i, j));
                let nx = dx/dist; let ny = dy/dist;
                let overlap = min_d - dist;
                let m1 = self.balls[i].mass; let m2 = self.balls[j].mass; let mt = m1+m2;
                self.balls[i].x += nx * overlap * (m2/mt);
                self.balls[i].y += ny * overlap * (m2/mt);
                self.balls[j].x -= nx * overlap * (m1/mt);
                self.balls[j].y -= ny * overlap * (m1/mt);
                let vn1 = self.balls[i].vx*nx + self.balls[i].vy*ny;
                let vn2 = self.balls[j].vx*nx + self.balls[j].vy*ny;
                let vn1n = (vn1*(m1-m2) + 2.0*m2*vn2)/mt;
                let vn2n = (vn2*(m2-m1) + 2.0*m1*vn1)/mt;
                self.balls[i].vx += (vn1n-vn1)*nx*self.bounce_coef;
                self.balls[i].vy += (vn1n-vn1)*ny*self.bounce_coef;
                self.balls[j].vx += (vn2n-vn2)*nx*self.bounce_coef;
                self.balls[j].vy += (vn2n-vn2)*ny*self.bounce_coef;
            }
        }}
        for (i, b) in self.balls.iter().enumerate() {
            self.trail[i].push((b.x, b.y));
            if self.trail[i].len() > self.trail_len { self.trail[i].remove(0); }
        }
        self.step_count += 1;
    }

    fn compute_reward(&self) -> f32 {
        if self.balls.is_empty() { return 0.0; }
        let p = &self.balls[self.player_idx];
        let dist = ((p.x-self.target.x).powi(2) + (p.y-self.target.y).powi(2)).sqrt();
        (-dist*5.0).exp() + if dist < self.target.radius*2.0 { 0.5 } else { 0.0 }
    }

    fn render(&self) -> image::RgbImage {
        let size = self.image_size as u32;
        let mut img = image::RgbImage::new(size, size);
        for y in 0..size { for x in 0..size {
            let v = (40.0 + (y as f32/size as f32)*15.0) as u8;
            img.put_pixel(x, y, image::Rgb([v,v,v]));
        }}
        for w in &self.walls {
            let x0 = ((w.x-w.w*0.5)*size as f32) as u32;
            let y0 = ((w.y-w.h*0.5)*size as f32) as u32;
            let x1 = ((w.x+w.w*0.5)*size as f32).min(size as f32) as u32;
            let y1 = ((w.y+w.h*0.5)*size as f32).min(size as f32) as u32;
            for py in y0..=y1 { for px in x0..=x1 {
                if px<size && py<size { img.put_pixel(px,py,image::Rgb([120,120,120])); }
            }}
        }
        let tx = (self.target.x*size as f32) as u32;
        let ty = (self.target.y*size as f32) as u32;
        let tr = (self.target.radius*size as f32) as u32;
        for py in ty.saturating_sub(tr)..=(ty+tr).min(size-1) {
            for px in tx.saturating_sub(tr)..=(tx+tr).min(size-1) {
                if ((px as i32-tx as i32).pow(2)+(py as i32-ty as i32).pow(2)) as u32 <= tr*tr {
                    img.put_pixel(px,py,image::Rgb([180,180,60]));
                }
            }
        }
        for (i, trail) in self.trail.iter().enumerate() {
            let (cr,cg,cb) = (self.balls[i].color[0],self.balls[i].color[1],self.balls[i].color[2]);
            for (k, &(tf_x, tf_y)) in trail.iter().enumerate() {
                let alpha = (k+1) as f32 / trail.len().max(1) as f32;
                let px = (tf_x*size as f32) as u32;
                let py = (tf_y*size as f32) as u32;
                let tr2 = (self.balls[i].radius*size as f32*0.4) as u32;
                for dy in -(tr2 as i32)..=(tr2 as i32) { for dx in -(tr2 as i32)..=(tr2 as i32) {
                    if dx*dx+dy*dy <= (tr2*tr2) as i32 {
                        let sx=(px as i32+dx)as u32; let sy=(py as i32+dy)as u32;
                        if sx<size && sy<size {
                            let e=img.get_pixel(sx,sy);
                            img.put_pixel(sx,sy,image::Rgb([
                                e[0].max((cr as f32*alpha) as u8),
                                e[1].max((cg as f32*alpha) as u8),
                                e[2].max((cb as f32*alpha) as u8)]));
                        }
                    }
                }}
            }
        }
        for b in &self.balls {
            let px=(b.x*size as f32)as u32; let py=(b.y*size as f32)as u32;
            let pr=(b.radius*size as f32)as u32;
            for dy in -(pr as i32)..=(pr as i32) { for dx in -(pr as i32)..=(pr as i32) {
                if dx*dx+dy*dy<=(pr*pr)as i32 {
                    let sx=(px as i32+dx)as u32; let sy=(py as i32+dy)as u32;
                    if sx<size && sy<size {
                        let d=((dx*dx+dy*dy)as f32).sqrt()/pr as f32;
                        let s=1.0-d*0.5;
                        img.put_pixel(sx,sy,image::Rgb([
                            (b.color[0]as f32*s)as u8,(b.color[1]as f32*s)as u8,(b.color[2]as f32*s)as u8]));
                    }
                }
            }}
        }
        img
    }

    fn to_tensor<B: Backend>(&self) -> Tensor<B, 3> {
        let img=self.render(); let s=self.image_size;
        let mut d=Vec::with_capacity(3*s*s);
        // NCHW planar: all R first, then all G, then all B
        for c in 0..3 {
            for y in 0..s { for x in 0..s {
                d.push(img.get_pixel(x as u32,y as u32)[c] as f32/255.0);
            }}
        }
        Tensor::<B,1>::from_floats(d.as_slice(),&Default::default()).reshape([3,s,s])
    }
}

impl Environment for ObjectWorld {
    fn reset<B: Backend>(&mut self, _device: &B::Device) -> Tensor<B, 3> {
        self.step_count=0; self.collision_events.clear();
        self.trail=vec![Vec::new();self.balls.len()];
        let balls_snap = self.balls.clone();
        for (i,b) in self.balls.iter_mut().enumerate() {
            loop {
                let x=0.15+self.rng.f32()*0.7;
                let y=0.1+self.rng.f32()*0.4;
                if balls_snap.iter().enumerate().all(|(j,o)| i==j || {
                    ((x-o.x).powi(2)+(y-o.y).powi(2)).sqrt() >= b.radius+o.radius+0.03
                }) { b.x=x; b.y=y; break; }
            }
            b.vx=(self.rng.f32()-0.5)*0.01; b.vy=(self.rng.f32()-0.5)*0.01;
        }
        loop {
            let tx=0.2+self.rng.f32()*0.6;
            let ty=0.75+self.rng.f32()*0.15;
            if self.balls.iter().all(|b| {
                ((tx-b.x).powi(2)+(ty-b.y).powi(2)).sqrt() >= b.radius+self.target.radius+0.02
            }) { self.target.x=tx; self.target.y=ty; break; }
        }
        self.to_tensor()
    }
    fn step<B: Backend>(&mut self, action: &[f32], _device: &B::Device) -> (Tensor<B, 3>, f32, bool) {
        let act = if action.is_empty() { 0.0 } else { action[0].clamp(-1.0,1.0) };
        self.physics_step(act);
        (self.to_tensor(), self.compute_reward(), self.step_count >= self.max_steps)
    }
    fn obs_shape(&self) -> [usize; 3] { [self.image_channels, self.image_size, self.image_size] }
    fn action_dim(&self) -> usize { 1 }
    fn max_steps(&self) -> usize { self.max_steps }
}
