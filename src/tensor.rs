use anyhow::Result;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub(crate) name: Arc<str>,
    pub(crate) dims: Arc<Vec<i64>>,
    pub(crate) data: Arc<Vec<f32>>,
}

impl Tensor {
    pub fn create(name: &str, dims: Vec<i64>, data: Vec<f32>) -> Result<Self> {
        let numel = dims
            .iter()
            .try_fold(1usize, |a, &d| {
                usize::try_from(d).ok().and_then(|d| a.checked_mul(d))
            })
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}': dims {dims:?} overflow"))?;
        if numel != data.len() {
            return Err(anyhow::anyhow!(
                "tensor '{name}': dims {dims:?} has {numel} elements, but data has {} elements",
                data.len()
            ));
        }
        Ok(Self {
            name: Arc::from(name),
            dims: Arc::new(dims),
            data: Arc::new(data),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn dims(&self) -> &[i64] {
        &self.dims
    }
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.dims.len()];
        for i in (0..self.dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1] as usize;
        }
        strides
    }
}
