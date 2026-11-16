use crate::tensor::Tensor;
use anyhow::{Result, anyhow};
use memmap2::Mmap;
use prost::Message;
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
static WEIGHTS: OnceLock<Arc<[Tensor]>> = OnceLock::new();

pub fn init_weights_from_file(path: impl AsRef<Path>) -> Result<()> {
    let mmap = unsafe { Mmap::map(&File::open(path.as_ref())?)? };
    let weights = decode_model(&mmap[..])?;
    WEIGHTS
        .set(Arc::from(weights))
        .map_err(|_| anyhow!("failed to init weights"))?;
    Ok(())
}

pub fn init_weights_from_bytes(bytes: &Arc<[u8]>) -> Result<()> {
    let weights = decode_model(bytes)?;
    WEIGHTS
        .set(Arc::from(weights))
        .map_err(|_| anyhow!("failed to init weights"))?;
    Ok(())
}

pub fn get_weights() -> Result<&'static Arc<[Tensor]>> {
    WEIGHTS
        .get()
        .ok_or_else(|| anyhow!("weights not initialized"))
}

pub fn decode_model(bytes: &[u8]) -> Result<Vec<Tensor>> {
    let model = ModelProto::decode(bytes)?;
    let graph = model
        .graph
        .ok_or_else(|| anyhow!("graph is missing from protobuf file"))?;
    let mut tensors = Vec::with_capacity(704); // ESRGAN has exact 704 initializers
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
                        "Tensor '{}': dims {:?} overflow element count",
                        tensor_proto.name,
                        tensor_proto.dims
                    )
                })?;
            if numel * size_of::<f32>() != tensor_proto.raw_data.len() {
                return Err(anyhow!(
                    "Tensor '{}': dims {:?} has {} elements, but raw_data has {} bytes",
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
        tensors.push(tensor);
    }
    Ok(tensors)
}

#[test]
fn test_init_weights() -> Result<()> {
    init_weights_from_file("esrgan-onnx-float/esrgan_single.onnx")?;
    let weights = get_weights()?;
    assert_eq!(weights.len(), 704);
    Ok(())
}
