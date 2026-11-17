use crate::kernels::*;
pub use crate::tensor::*;
use anyhow::{Result, anyhow};
use memmap2::Mmap;
use prost::Message;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, OnceLock};

mod kernels;
mod tensor;

#[derive(Clone, PartialEq, Message)]
struct ModelProto {
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
}

#[derive(Clone, PartialEq, Message)]
struct GraphProto {
    #[prost(message, repeated, tag = "5")]
    initializer: Vec<TensorProto>,
}

#[derive(Clone, PartialEq, Message)]
struct TensorProto {
    #[prost(string, tag = "8")]
    name: String,
    #[prost(int64, repeated, tag = "1")]
    dims: Vec<i64>,
    #[prost(float, repeated, tag = "4")]
    float_data: Vec<f32>,
    #[prost(bytes = "vec", tag = "9")]
    raw_data: Vec<u8>,
}
static WEIGHTS: OnceLock<HashMap<String, Tensor>> = OnceLock::new();

fn decode_model(bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
    let model = ModelProto::decode(bytes)?;
    let graph = model
        .graph
        .ok_or_else(|| anyhow!("graph is missing from protobuf file"))?;

    // ESRGAN has exact 704 initializers
    let mut tensors = HashMap::with_capacity(704);
    for tensor_proto in graph.initializer {
        let tensor_data = if !tensor_proto.raw_data.is_empty() {
            let numel: usize = tensor_proto
                .dims
                .iter()
                .try_fold(1usize, |a, &d| {
                    usize::try_from(d).ok().and_then(|d| a.checked_mul(d))
                })
                .ok_or_else(|| {
                    anyhow!(
                        "tensor '{}': dims {:?} overflow element count",
                        tensor_proto.name,
                        tensor_proto.dims
                    )
                })?;
            if numel * size_of::<f32>() != tensor_proto.raw_data.len() {
                return Err(anyhow!(
                    "tensor '{}': dims {:?} has {} elements, but raw_data has {} bytes",
                    tensor_proto.name,
                    tensor_proto.dims,
                    numel,
                    tensor_proto.raw_data.len()
                ));
            }

            let (bytes_chunks, _) = tensor_proto.raw_data.as_chunks::<4>();
            let mut data = Vec::with_capacity(bytes_chunks.len());
            data.extend(bytes_chunks.iter().map(|c| f32::from_le_bytes(*c)));
            data
        } else {
            tensor_proto.float_data
        };

        let tensor = Tensor::create(&tensor_proto.name, tensor_proto.dims, tensor_data)?;
        tensors.insert(tensor_proto.name, tensor);
    }
    Ok(tensors)
}

pub fn init_weights_from_file(path: impl AsRef<Path>) -> Result<()> {
    let mmap = unsafe { Mmap::map(&File::open(path.as_ref())?)? };
    let weights = decode_model(&mmap[..])?;
    WEIGHTS
        .set(weights)
        .map_err(|_| anyhow!("failed to init weights"))?;
    Ok(())
}

pub fn init_weights_from_bytes(bytes: &Arc<[u8]>) -> Result<()> {
    let weights = decode_model(bytes)?;
    WEIGHTS
        .set(weights)
        .map_err(|_| anyhow!("failed to init weights"))?;
    Ok(())
}

pub fn get_weights() -> Result<&'static HashMap<String, Tensor>> {
    WEIGHTS
        .get()
        .ok_or_else(|| anyhow!("weights not initialized"))
}

pub fn run(input: &Tensor) -> Result<Tensor> {
    let weights = get_weights()?;
    let w = |name: &str| -> Result<&Tensor> {
        weights
            .get(name)
            .ok_or_else(|| anyhow!("weight '{name}' not found"))
    };

    // one RDB: 4 dense leaky convs with concat growth 64→96→128→160→192,
    // then conv5 (no leaky) and a PLAIN residual add
    let rdb = |x: &Tensor, block: usize, rdb_idx: usize| -> Result<Tensor> {
        let r = rdb_idx + 1;
        // conv5 weights are in form of val_{7+6k} / val_{8+6k}
        let k = block * 3 + rdb_idx;

        let c1 = leaky_relu(
            &conv(
                x,
                w(&format!("model.RRDB_trunk.{block}.RDB{r}.conv1.weight"))?,
                w(&format!("model.RRDB_trunk.{block}.RDB{r}.conv1.bias"))?,
                &format!("rrdb{block}_rdb{r}_conv1"),
            )?,
            &format!("rrdb{block}_rdb{r}_l1"),
        )?;
        let cat1 = concat(&[x, &c1], &format!("rrdb{block}_rdb{r}_cat1"))?; // 64+32=96

        let c2 = leaky_relu(
            &conv(
                &cat1,
                w(&format!("model.RRDB_trunk.{block}.RDB{r}.conv2.weight"))?,
                w(&format!("model.RRDB_trunk.{block}.RDB{r}.conv2.bias"))?,
                &format!("rrdb{block}_rdb{r}_conv2"),
            )?,
            &format!("rrdb{block}_rdb{r}_l2"),
        )?;
        let cat2 = concat(&[x, &c1, &c2], &format!("rrdb{block}_rdb{r}_cat2"))?; // 64+32+32=128

        let c3 = leaky_relu(
            &conv(
                &cat2,
                w(&format!("model.RRDB_trunk.{block}.RDB{r}.conv3.weight"))?,
                w(&format!("model.RRDB_trunk.{block}.RDB{r}.conv3.bias"))?,
                &format!("rrdb{block}_rdb{r}_conv3"),
            )?,
            &format!("rrdb{block}_rdb{r}_l3"),
        )?;
        let cat3 = concat(&[x, &c1, &c2, &c3], &format!("rrdb{block}_rdb{r}_cat3"))?; // 64+32*3=160

        let c4 = leaky_relu(
            &conv(
                &cat3,
                w(&format!("model.RRDB_trunk.{block}.RDB{r}.conv4.weight"))?,
                w(&format!("model.RRDB_trunk.{block}.RDB{r}.conv4.bias"))?,
                &format!("rrdb{block}_rdb{r}_conv4"),
            )?,
            &format!("rrdb{block}_rdb{r}_l4"),
        )?;
        let cat4 = concat(
            &[x, &c1, &c2, &c3, &c4],
            &format!("rrdb{block}_rdb{r}_cat4"),
        )?; // 64+32*4=192

        let c5 = conv(
            &cat4,
            w(&format!("val_{}", 7 + 6 * k))?,
            w(&format!("val_{}", 8 + 6 * k))?,
            &format!("rrdb{block}_rdb{r}_conv5"),
        )?;

        // residual add(conv5_out, rdb_in) in the end, no leaky
        add(&c5, x, &format!("rrdb{block}_rdb{r}_add"))
    };

    // one RRDB: 3 RDBs chained, then the ONLY 0.2 mul (val_0, which is constant here) + residual add
    let rrdb_block = |x: &Tensor, block: usize| -> Result<Tensor> {
        let r1 = rdb(x, block, 0)?;
        let r2 = rdb(&r1, block, 1)?;
        let r3 = rdb(&r2, block, 2)?;
        let scaled = mul(&r3, w("val_0")?, &format!("rrdb{block}_mul"))?;
        add(&scaled, x, &format!("rrdb{block}_add"))
    };

    // first_conv
    let first = conv(
        input,
        w("model.conv_first.weight")?,
        w("model.conv_first.bias")?,
        "conv_first",
    )?;

    // ESRGAN has 23 RRDB blocks
    let mut x = rrdb_block(&first, 0)?;
    for block in 1..23 {
        x = rrdb_block(&x, block)?;
    }

    // end conv after RRDB blocks
    // out = trunk_conv(trunk_out) + first
    let trunk = conv(
        &x,
        w("model.trunk_conv.weight")?,
        w("model.trunk_conv.bias")?,
        "trunk_conv",
    )?;
    x = add(&trunk, &first, "trunk_add")?;

    // two upscaling, (nearest, [1,1,2,2]) -> conv -> leaky

    x = resize(&x, w("val_1")?, "upsample1")?;
    x = leaky_relu(
        &conv(
            &x,
            w("model.upconv1.weight")?,
            w("model.upconv1.bias")?,
            "upconv1",
        )?,
        "upconv1_l",
    )?;
    x = resize(&x, w("val_1")?, "upsample2")?;

    // last two leaky convs, then conv_last (no leaky) for final output
    x = leaky_relu(
        &conv(
            &x,
            w("model.upconv2.weight")?,
            w("model.upconv2.bias")?,
            "upconv2",
        )?,
        "upconv2_l",
    )?;
    x = leaky_relu(
        &conv(
            &x,
            w("model.HRconv.weight")?,
            w("model.HRconv.bias")?,
            "HRconv",
        )?,
        "HRconv_l",
    )?;

    let output = conv(
        &x,
        w("model.conv_last.weight")?,
        w("model.conv_last.bias")?,
        "conv_last",
    )?;
    Ok(output)
}

#[test]
fn test_init_weights() {
    init_weights_from_file("esrgan-onnx-float/esrgan_single.onnx").unwrap();
    let weights = get_weights().unwrap();
    assert_eq!(weights.len(), 704);
}

#[test]
fn test_image_upscale() {
    init_weights_from_file("esrgan-onnx-float/esrgan_single.onnx").unwrap();
    let _ = get_weights().unwrap();

    let width = 128;
    let height = 128;
    let plane = width * height;
    let mut data = vec![0f32; 3 * plane];
    let in_image = image::open("input.png").unwrap().to_rgb8();
    for y in 0..height {
        for x in 0..width {
            let pixel = in_image.get_pixel(x as u32, y as u32);
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            data[(y * width + x) * 3] = r;
            data[(y * width + x) * 3 + 1] = g;
            data[(y * width + x) * 3 + 2] = b;
        }
    }

    // run inference
    let input = Tensor::create("inputs", vec![1, 3, 128, 128], data).unwrap();
    let output = run(&input).unwrap();
    assert_eq!(output.dims(), &[1, 3, 512, 512]);
}
