use crate::kernels::*;
pub use crate::tensor::*;
use anyhow::{Result, anyhow};
use prost::Message;
use rustc_hash::FxHashMap;
use std::path::Path;
use std::sync::OnceLock;

mod kernels;
mod tensor;

static GLOBAL_THREAD_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
static INITIALIZERS: OnceLock<FxHashMap<String, Tensor>> = OnceLock::new();
fn decode_model(bytes: &[u8]) -> Result<FxHashMap<String, Tensor>> {
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

    {
        use sha2::Digest;

        const CHECKSUM: &str = "5a365f595282abe413e80a77fc58d0b00064c61fa55c9740cfc3d6c1ac0c5a4f";
        let mut hasher = sha2::Sha256::new();

        let (chunk_f, chunk_l) = bytes.as_chunks::<4096>();
        for c in chunk_f {
            hasher.update(c);
        }
        hasher.update(chunk_l);

        let checksum = hex::encode(hasher.finalize().as_slice());
        if CHECKSUM != checksum {
            return Err(anyhow!(
                "model checksum mismatch, expected {CHECKSUM}, got {checksum}"
            ));
        }
    }

    let model = ModelProto::decode(bytes)?;
    let graph = model
        .graph
        .ok_or_else(|| anyhow!("graph is missing from protobuf file"))?;

    // ESRGAN has exact 704 initializers
    let mut tensors = FxHashMap::with_capacity_and_hasher(704, Default::default());
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

    // init rayon thread pool
    GLOBAL_THREAD_POOL
        .set(rayon::ThreadPoolBuilder::new().build()?)
        .map_err(|_| anyhow!("failed to init global thread pool"))?;

    Ok(tensors)
}

pub fn load_model_from_file(path: impl AsRef<Path>) -> Result<()> {
    let mmap = unsafe { memmap2::Mmap::map(&std::fs::File::open(path.as_ref())?)? };
    let weights = decode_model(&mmap[..])?;
    INITIALIZERS
        .set(weights)
        .map_err(|_| anyhow!("failed to init initializers"))?;
    Ok(())
}

pub fn load_model_from_bytes(bytes: impl AsRef<[u8]>) -> Result<()> {
    let weights = decode_model(bytes.as_ref())?;
    INITIALIZERS
        .set(weights)
        .map_err(|_| anyhow!("failed to init initializers"))?;
    Ok(())
}

pub fn run(input: &Tensor) -> Result<Tensor> {
    let weights = INITIALIZERS
        .get()
        .ok_or_else(|| anyhow!("model not loaded yet"))?;

    let w = |name: &str| -> Result<&Tensor> {
        weights
            .get(name)
            .ok_or_else(|| anyhow!("initializer '{name}' not found"))
    };

    // allocate an output buffer for a conv given its input + weight dims
    let conv_out = |input: &Tensor, weight: &Tensor, name: &str| -> Result<Tensor> {
        let in_dims = &input.dims;
        let w_dims = &weight.dims;
        let n = in_dims[0] as usize;
        let cout = w_dims[0] as usize;
        let h = in_dims[2] as usize;
        let wdt = in_dims[3] as usize;
        Tensor::create(
            name,
            vec![n as i64, cout as i64, h as i64, wdt as i64],
            vec![0f32; n * cout * h * wdt],
        )
    };

    // one RDB: 4 dense leaky convs with concat growth 64→96→128→160→192,
    // then conv5 (no leaky) and a PLAIN residual add
    let rdb = |x: &Tensor, block: usize, rdb_idx: usize| -> Result<Tensor> {
        let r = rdb_idx + 1;
        // conv5 weights are in form of val_{7+6k} / val_{8+6k}
        let k = block * 3 + rdb_idx;

        let w1 = w(&format!("model.RRDB_trunk.{block}.RDB{r}.conv1.weight"))?;
        let b1 = w(&format!("model.RRDB_trunk.{block}.RDB{r}.conv1.bias"))?;
        let mut c1 = conv_out(x, w1, &format!("rrdb{block}_rdb{r}_conv1"))?;
        conv(x, w1, b1, &mut c1, true)?;
        let cat1 = concat(&[x, &c1], &format!("rrdb{block}_rdb{r}_cat1"))?; // 64+32=96

        let w2 = w(&format!("model.RRDB_trunk.{block}.RDB{r}.conv2.weight"))?;
        let b2 = w(&format!("model.RRDB_trunk.{block}.RDB{r}.conv2.bias"))?;
        let mut c2 = conv_out(&cat1, w2, &format!("rrdb{block}_rdb{r}_conv2"))?;
        conv(&cat1, w2, b2, &mut c2, true)?;
        let cat2 = concat(&[x, &c1, &c2], &format!("rrdb{block}_rdb{r}_cat2"))?; // 64+32+32=128

        let w3 = w(&format!("model.RRDB_trunk.{block}.RDB{r}.conv3.weight"))?;
        let b3 = w(&format!("model.RRDB_trunk.{block}.RDB{r}.conv3.bias"))?;
        let mut c3 = conv_out(&cat2, w3, &format!("rrdb{block}_rdb{r}_conv3"))?;
        conv(&cat2, w3, b3, &mut c3, true)?;
        let cat3 = concat(&[x, &c1, &c2, &c3], &format!("rrdb{block}_rdb{r}_cat3"))?; // 64+32*3=160

        let w4 = w(&format!("model.RRDB_trunk.{block}.RDB{r}.conv4.weight"))?;
        let b4 = w(&format!("model.RRDB_trunk.{block}.RDB{r}.conv4.bias"))?;
        let mut c4 = conv_out(&cat3, w4, &format!("rrdb{block}_rdb{r}_conv4"))?;
        conv(&cat3, w4, b4, &mut c4, true)?;
        let cat4 = concat(
            &[x, &c1, &c2, &c3, &c4],
            &format!("rrdb{block}_rdb{r}_cat4"),
        )?; // 64+32*4=192

        let w5 = w(&format!("val_{}", 7 + 6 * k))?;
        let b5 = w(&format!("val_{}", 8 + 6 * k))?;
        let mut c5 = conv_out(&cat4, w5, &format!("rrdb{block}_rdb{r}_conv5"))?;
        conv(&cat4, w5, b5, &mut c5, false)?;

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
    let wf = w("model.conv_first.weight")?;
    let bf = w("model.conv_first.bias")?;
    let mut first = conv_out(input, wf, "conv_first")?;
    conv(input, wf, bf, &mut first, false)?;

    // ESRGAN has 23 RRDB blocks
    let mut x = rrdb_block(&first, 0)?;
    for block in 1..23 {
        x = rrdb_block(&x, block)?;
    }

    // end conv after RRDB blocks
    // out = trunk_conv(trunk_out) + first
    let wt = w("model.trunk_conv.weight")?;
    let bt = w("model.trunk_conv.bias")?;
    let mut trunk = conv_out(&x, wt, "trunk_conv")?;
    conv(&x, wt, bt, &mut trunk, false)?;
    x = add(&trunk, &first, "trunk_add")?;

    // two upscaling, (nearest, [1,1,2,2]) -> conv -> leaky

    x = resize(&x, w("val_1")?, "upsample1")?;
    let wu1 = w("model.upconv1.weight")?;
    let bu1 = w("model.upconv1.bias")?;
    let mut up1 = conv_out(&x, wu1, "upconv1")?;
    conv(&x, wu1, bu1, &mut up1, true)?;
    x = up1;

    x = resize(&x, w("val_1")?, "upsample2")?;
    let wu2 = w("model.upconv2.weight")?;
    let bu2 = w("model.upconv2.bias")?;
    let mut up2 = conv_out(&x, wu2, "upconv2")?;
    conv(&x, wu2, bu2, &mut up2, true)?;
    x = up2;

    // last two leaky convs, then conv_last (no leaky) for final output
    let wh = w("model.HRconv.weight")?;
    let bh = w("model.HRconv.bias")?;
    let mut hr = conv_out(&x, wh, "HRconv")?;
    conv(&x, wh, bh, &mut hr, true)?;
    x = hr;

    let wl = w("model.conv_last.weight")?;
    let bl = w("model.conv_last.bias")?;
    let mut output = conv_out(&x, wl, "conv_last")?;
    conv(&x, wl, bl, &mut output, false)?;
    Ok(output)
}

#[test]
fn test_load_model() {
    load_model_from_file("esrgan-onnx-float/esrgan_single.onnx").unwrap();
    let weights = INITIALIZERS.get().unwrap();
    assert_eq!(weights.len(), 704);
}

#[test]
fn test_image_upscale() {
    load_model_from_file("esrgan-onnx-float/esrgan_single.onnx").unwrap();

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
