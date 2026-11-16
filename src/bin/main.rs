use anyhow::Result;
use esrgan_rs::init_weights_from_file;

fn main() -> Result<()> {
    init_weights_from_file("esrgan-onnx-float/esrgan_single.onnx")?;
    Ok(())
}
