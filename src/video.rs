use burn::tensor::{backend::Backend, Tensor};
use std::path::Path;
use std::process::Command;

/// Convert a Burn image tensor [C, H, W] (NCHW, values in [0,1]) to an RGB image.
fn tensor_to_image<B: Backend>(tensor: &Tensor<B, 3>) -> image::RgbImage {
    let data = tensor.to_data();
    let shape = &data.shape;
    let h = shape[1];
    let w = shape[2];
    let values = data.as_slice::<f32>().unwrap();
    let mut img = image::RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx_r = 0 * h * w + y * w + x;
            let idx_g = 1 * h * w + y * w + x;
            let idx_b = 2 * h * w + y * w + x;
            let r = (values[idx_r] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (values[idx_g] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (values[idx_b] * 255.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }
    img
}

/// Save a Burn image tensor as a PNG file.
pub fn save_frame<B: Backend>(tensor: &Tensor<B, 3>, path: &str) {
    let img = tensor_to_image::<B>(tensor);
    img.save(path).expect("Failed to save frame");
}

/// Make a side-by-side comparison frame: [real || recon].
pub fn make_comparison_frame<B: Backend>(
    real: &Tensor<B, 3>,
    recon: &Tensor<B, 3>,
) -> Tensor<B, 3> {
    let real_data = real.to_data();
    let recon_data = recon.to_data();
    let shape = &real_data.shape;
    let c = shape[0];
    let h = shape[1];
    let w = shape[2];
    let real_vals = real_data.as_slice::<f32>().unwrap();
    let recon_vals = recon_data.as_slice::<f32>().unwrap();

    let w2 = 2 * w;
    let mut img = image::RgbImage::new(w2 as u32, h as u32);

    for y in 0..h {
        for x in 0..w {
            let get_pixel = |vals: &[f32], cx: usize, cy: usize, chan: usize| -> u8 {
                let idx = chan * h * w + cy * w + cx;
                (vals[idx] * 255.0).clamp(0.0, 255.0) as u8
            };

            // Left: real
            img.put_pixel(x as u32, y as u32, image::Rgb([
                get_pixel(real_vals, x, y, 0),
                get_pixel(real_vals, x, y, 1),
                get_pixel(real_vals, x, y, 2),
            ]));
            // Right: recon
            img.put_pixel((w + x) as u32, y as u32, image::Rgb([
                get_pixel(recon_vals, x, y, 0),
                get_pixel(recon_vals, x, y, 1),
                get_pixel(recon_vals, x, y, 2),
            ]));
        }
    }

    // Convert back to Burn tensor [C, H, 2W] in NCHW order
    let total = c * h * w2;
    let mut data = vec![0.0f32; total];
    for chan in 0..c {
        for y in 0..h {
            for x in 0..w2 {
                let pixel = img.get_pixel(x as u32, y as u32);
                data[chan * h * w2 + y * w2 + x] = pixel[chan] as f32 / 255.0;
            }
        }
    }
    let device = real.device();
    Tensor::<B, 1>::from_floats(data.as_slice(), &device).reshape([c, h, w2])
}

/// Combine PNG frames in a directory into an MP4 video using ffmpeg.
pub fn frames_to_mp4(frame_dir: &str, output_path: &str, fps: u32) {
    let dir = Path::new(frame_dir);
    if !dir.exists() {
        std::fs::create_dir_all(dir).expect("Failed to create frame dir");
    }

    if let Some(parent) = Path::new(output_path).parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).expect("Failed to create output dir");
        }
    }

    let pattern = format!("{}/frame_%04d.png", frame_dir);
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate",
            &fps.to_string(),
            "-i",
            &pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ])
        .status();

    match status {
        Ok(exit) if exit.success() => {
            println!("Video saved to: {}", output_path);
        }
        Ok(exit) => {
            eprintln!("ffmpeg failed with exit code: {:?}", exit.code());
            eprintln!("Frames are in: {}/ — you can combine them manually.", frame_dir);
        }
        Err(e) => {
            eprintln!("ffmpeg not found ({}). Install ffmpeg or combine frames manually.", e);
            eprintln!("Frames saved to: {}/", frame_dir);
        }
    }
}
