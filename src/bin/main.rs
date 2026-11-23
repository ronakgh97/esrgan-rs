use anyhow::Result;
use clap::Parser;
use esrgan_rs::{Tensor, load_model_from_bytes, run};
use image::{GenericImageView, RgbImage};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;

#[derive(Parser)]
#[command(
    name = "esrgan",
    author = "ronakgh97 <ronakgh97@gmail.com>",
    version = "v1.0.0",
    about = "AI Upscaler using Original ESRGAN model",
    long_about = "AI Upscaler using Original ESRGAN model"
)]
struct Cli {
    /// Input image path
    #[arg(short, long)]
    input: PathBuf,
    /// Output image path
    #[arg(short, long)]
    output: PathBuf,
}

const WEIGHT_BYTES: &[u8] = include_bytes!("../../esrgan-onnx-float/esrgan_single.onnx");

fn main() -> Result<()> {
    load_model_from_bytes(WEIGHT_BYTES)?;

    let args = Cli::parse();
    let in_img = image::ImageReader::open(&args.input)?
        .with_guessed_format()?
        .decode()?
        .into_rgb8();

    const INPUT_SCALE: usize = 128;
    const OUTPUT_SCALE: usize = 512;
    const SCALE_FACTOR: usize = OUTPUT_SCALE / INPUT_SCALE;

    let (w, h) = (in_img.width() as usize, in_img.height() as usize);
    let tile_nx = w.div_ceil(INPUT_SCALE) * INPUT_SCALE;
    let tile_ny = h.div_ceil(INPUT_SCALE) * INPUT_SCALE;

    let mut out_img = RgbImage::from_raw(
        (w * SCALE_FACTOR) as u32,
        (h * SCALE_FACTOR) as u32,
        vec![0u8; (w * SCALE_FACTOR) * (h * SCALE_FACTOR) * 3],
    )
    .ok_or_else(|| anyhow::anyhow!("unable to create output image buffer"))?;

    // process each tile of the input image in 128x128,
    // run the inference and upscale each to 512x512,
    // write all tiles to the output image
    println!(
        "upscaling {}x{} image, to {}x{} output",
        w,
        h,
        w * SCALE_FACTOR,
        h * SCALE_FACTOR,
    );

    let start = std::time::Instant::now();
    let progress = AtomicU64::new(0);
    let total_progress = (tile_nx / INPUT_SCALE) * (tile_ny / INPUT_SCALE);
    let input_tile_scale = INPUT_SCALE * INPUT_SCALE;
    let mut tensor_data = vec![0f32; 3 * input_tile_scale];
    for ty in (0..tile_ny).step_by(INPUT_SCALE) {
        for tx in (0..tile_nx).step_by(INPUT_SCALE) {
            tensor_data.fill(0.0); // reuse the buffer
            // process each tiles
            for y in 0..INPUT_SCALE {
                for x in 0..INPUT_SCALE {
                    // get the pixel from image tiles
                    let (px, py) = (tx + x, ty + y);
                    // check if the pixel is within the bounds of the original image
                    if px < w && py < h {
                        let p = unsafe { in_img.unsafe_get_pixel(px as u32, py as u32) };
                        let r = p[0] as f32 / 255.0;
                        let g = p[1] as f32 / 255.0;
                        let b = p[2] as f32 / 255.0;
                        tensor_data[y * INPUT_SCALE + x] = r;
                        tensor_data[input_tile_scale + y * INPUT_SCALE + x] = g;
                        tensor_data[2 * input_tile_scale + y * INPUT_SCALE + x] = b;
                    }
                }
            }

            // run the inference here
            let input_tensor = Tensor::create(
                "inputs",
                vec![1, 3, INPUT_SCALE as i64, INPUT_SCALE as i64],
                tensor_data.clone(),
            )?;
            let output = run(&input_tensor)?;
            let upscaled = output.data();

            progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            std::io::stdout().flush()?;
            print!(
                "\rprogressing: {}/{} tiles processed",
                progress.load(std::sync::atomic::Ordering::Relaxed),
                total_progress
            );

            // extract the red, green, and blue planes from the upscaled output tile
            let output_tile_scale = OUTPUT_SCALE * OUTPUT_SCALE;
            let red_plane = &upscaled[..output_tile_scale];
            let green_plane = &upscaled[output_tile_scale..2 * output_tile_scale];
            let blue_plane = &upscaled[2 * output_tile_scale..];

            // how much of this 512x512 tile actually fits inside the real image
            let (tile_output_x, tile_output_y) = (tx * SCALE_FACTOR, ty * SCALE_FACTOR);
            let (visible_width, visible_height) = (
                (w * SCALE_FACTOR - tile_output_x).min(OUTPUT_SCALE),
                (h * SCALE_FACTOR - tile_output_y).min(OUTPUT_SCALE),
            );

            // write the upscaled tile, clipped to the real output bounds
            let output_buf: &mut [u8] = &mut out_img;
            let output_row_len = w * SCALE_FACTOR * 3;
            for y in 0..visible_height {
                // calculate the offset in the upscaled tile for the current row
                let y_offset = y * OUTPUT_SCALE;

                // get the corresponding rows from the red, green, and blue planes
                let r_row = &red_plane[y_offset..y_offset + visible_width];
                let g_row = &green_plane[y_offset..y_offset + visible_width];
                let b_row = &blue_plane[y_offset..y_offset + visible_width];

                // calculate the starting index, and get the output buffer image for the current row
                let output_row_start = (tile_output_y + y) * output_row_len + tile_output_x * 3; // y_height * a + x_width
                let output_row =
                    &mut output_buf[output_row_start..output_row_start + visible_width * 3];
                for x in 0..visible_width {
                    // write the visible pixels only
                    output_row[x * 3] = (r_row[x] * 255.0) as u8;
                    output_row[x * 3 + 1] = (g_row[x] * 255.0) as u8;
                    output_row[x * 3 + 2] = (b_row[x] * 255.0) as u8;
                }
            }
        }
    }

    println!();
    println!(
        "finished processing all tiles, took {:.2}s",
        start.elapsed().as_secs_f32()
    );
    out_img.save(&args.output)?;

    Ok(())
}
