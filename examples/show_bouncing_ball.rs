//! Quick demo: render bouncing_ball frames and save as PNGs.
//! Run: cargo run --example show_bouncing_ball

use std::path::Path;

fn main() {
    // Simulate the bouncing ball physics and rendering directly
    // (no Burn dependency — pure CPU rendering)

    let image_size: usize = 256;
    let image_channels: usize = 3;
    let max_steps: usize = 200;

    // Same physics as BouncingBall
    let mut x: f32 = 0.5;
    let mut y: f32 = 0.7;
    let mut vx: f32 = 0.005;
    let mut vy: f32 = 0.0;
    let gravity: f32 = 0.002;
    let bounce_damping: f32 = 0.8;
    let control_force: f32 = 0.001;
    let force_x: f32 = 0.3; // constant rightward force

    let out_dir = "frames/bouncing_ball_demo";
    std::fs::create_dir_all(out_dir).expect("create dir");

    for step in 0..max_steps {
        // physics
        vx += force_x * control_force;
        vy -= gravity;
        x += vx;
        y += vy;

        if x < 0.0 { x = 0.0; vx = -vx * bounce_damping; }
        if x > 1.0 { x = 1.0; vx = -vx * bounce_damping; }
        if y < 0.0 { y = 0.0; vy = -vy * bounce_damping; }
        if y > 1.0 { y = 1.0; vy = -vy * bounce_damping; }

        vx *= 0.99;
        vy *= 0.99;

        // render CHW
        let pixels = render_chw(x, y, vx, vy, image_size, image_channels);

        // save as PNG (convert CHW → RGB image)
        save_chw_png(&pixels, image_size, image_channels, &format!("{}/frame_{:04}.png", out_dir, step));
    }

    println!("Saved {} frames to {}/", max_steps, out_dir);
    println!("Run: ffmpeg -y -framerate 30 -i {}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p bouncing_ball_demo.mp4", out_dir);
}

fn render_chw(x: f32, y: f32, vx: f32, vy: f32, size: usize, channels: usize) -> Vec<f32> {
    let n = size * size;
    let mut p = vec![0.0f32; channels * n];

    // sky gradient
    for py in 0..size {
        let sky = (size - py) as f32 / size as f32;
        for px in 0..size {
            let dst = py * size + px;
            p[dst] = sky * 0.3;
            p[n + dst] = sky * 0.6;
            p[2 * n + dst] = sky * 0.9;
        }
    }

    // ground
    let gy = size - 1;
    for px in 0..size {
        let dst = gy * size + px;
        p[dst] = 0.2;
        p[n + dst] = 0.8;
        p[2 * n + dst] = 0.2;
    }

    // ball
    let bx = (x * size as f32) as usize;
    let by = ((1.0 - y) * size as f32) as usize;
    let br = (size / 16).max(2);
    for dy in -(br as i32)..=(br as i32) {
        for dx in -(br as i32)..=(br as i32) {
            if dx * dx + dy * dy <= (br * br) as i32 {
                let px = (bx as i32 + dx).clamp(0, size as i32 - 1) as usize;
                let py = (by as i32 + dy).clamp(0, size as i32 - 1) as usize;
                let dst = py * size + px;
                let dr = ((dx * dx + dy * dy) as f32).sqrt() / br as f32;
                let hl = (1.0 - dr).max(0.0);
                p[dst] = 0.9 + hl * 0.1;
                p[n + dst] = 0.1 + hl * 0.5;
                p[2 * n + dst] = 0.1 + hl * 0.3;
            }
        }
    }

    // velocity arrow (Bresenham)
    if vx.abs() > 0.001 || vy.abs() > 0.001 {
        let al = 20.min(size / 8);
        let ex = bx as i32 + (vx * al as f32 * 40.0) as i32;
        let ey = by as i32 - (vy * al as f32 * 40.0) as i32;
        let ex = ex.clamp(0, size as i32 - 1) as usize;
        let ey = ey.clamp(0, size as i32 - 1) as usize;
        draw_line_chw(&mut p, bx, by, ex, ey, size, n, [1.0, 1.0, 0.0]);
    }

    p
}

fn draw_line_chw(p: &mut [f32], x0: usize, y0: usize, x1: usize, y1: usize, size: usize, n: usize, color: [f32; 3]) {
    let dx = (x1 as i32 - x0 as i32).abs();
    let dy = (y1 as i32 - y0 as i32).abs();
    let sx: i32 = if x0 < x1 { 1 } else { -1 };
    let sy: i32 = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    let mut x = x0 as i32;
    let mut y = y0 as i32;
    for _ in 0..200 {
        if x >= 0 && x < size as i32 && y >= 0 && y < size as i32 {
            let dst = (y as usize) * size + (x as usize);
            p[dst] = color[0];
            p[n + dst] = color[1];
            p[2 * n + dst] = color[2];
        }
        if x == x1 as i32 && y == y1 as i32 { break; }
        let e2 = 2 * err;
        if e2 > -dy { err -= dy; x += sx; }
        if e2 < dx { err += dx; y += sy; }
    }
}

fn save_chw_png(pixels: &[f32], size: usize, channels: usize, path: &str) {
    let n = size * size;
    let mut img = image::RgbImage::new(size as u32, size as u32);
    for py in 0..size {
        for px in 0..size {
            let dst = py * size + px;
            let r = (pixels[dst] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (pixels[n + dst] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (pixels[2 * n + dst] * 255.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(px as u32, py as u32, image::Rgb([r, g, b]));
        }
    }
    img.save(Path::new(path)).expect("save png");
}
