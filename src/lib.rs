use crate::tensor::Tensor;
use prost::Message;
use std::path::Path;
use std::sync::{Arc, OnceLock};

mod kernels;
mod tensor;

use anyhow::Result;

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
    let mmap = unsafe { memmap2::Mmap::map(&std::fs::File::open(path.as_ref())?)? };
    let weights = decode_model(&mmap[..])?;
    WEIGHTS
        .set(Arc::from(weights))
        .map_err(|_| anyhow::anyhow!("failed to init weights"))?;
    Ok(())
}

pub fn init_weights_from_bytes(bytes: &Arc<[u8]>) -> Result<()> {
    let weights = decode_model(bytes)?;
    WEIGHTS
        .set(Arc::from(weights))
        .map_err(|_| anyhow::anyhow!("failed to init weights"))?;
    Ok(())
}

pub fn get_weights() -> Result<&'static Arc<[Tensor]>> {
    WEIGHTS
        .get()
        .ok_or_else(|| anyhow::anyhow!("weights not initialized"))
}

pub fn decode_model(bytes: &[u8]) -> Result<Vec<Tensor>> {
    let model = ModelProto::decode(bytes)?;
    let graph = model
        .graph
        .ok_or_else(|| anyhow::anyhow!("graph is missing from protobuf file"))?;
    let mut tensors = Vec::with_capacity(704); // ESRGAN has exact 704 initializers
    for tensor_proto in graph.initializer {
        let tensor = Tensor::create(
            &tensor_proto.name,
            tensor_proto.dims,
            tensor_proto.float_data,
        )?;
        tensors.push(tensor);
    }
    Ok(tensors)
}
