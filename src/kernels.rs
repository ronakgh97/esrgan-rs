use crate::tensor::Tensor;
use anyhow::Result;

pub fn conv(input: &Tensor, weight: &Tensor, bias: &Tensor, name: &str) -> Result<Tensor> {
    // ESRGAN hardcoded attributes
    let _auto_pad = "NOTSET";
    let _group = 1;
    let _dilations = [1, 1];
    let _pads = [1, 1, 1, 1];
    let _strides = [1, 1];
    todo!()
}

pub fn leaky_relu(input: &Tensor, name: &str) -> Result<Tensor> {
    // ESRGAN hardcoded attributes
    const ALPHA: f32 = 0.2;

    let data = input.data();
    let output_data: Vec<f32> = data
        .iter()
        .map(|&x| if x < 0.0 { x * ALPHA } else { x })
        .collect();
    Tensor::create(name, input.dims().to_vec(), output_data)
}

pub fn concat(input_a: &Tensor, input_b: &Tensor, name: &str) -> Result<Tensor> {
    // ESRGAN hardcoded attributes
    let _axis = 1;
    todo!()
}
pub fn add(input_a: &Tensor, input_b: &Tensor, name: &str) -> Result<Tensor> {
    todo!()
}
pub fn mul(input_a: &Tensor, input_b: &Tensor, name: &str) -> Result<Tensor> {
    todo!()
}
pub fn resize(input: &Tensor, name: &str) -> Result<Tensor> {
    // ESRGAN hardcoded attributes
    let _antialias = 0;
    let _coordinate_transformation_mode = "asymmetric";
    let _cubic_coeff_a = -0.75;
    let _exclude_outside = 0;
    let _extrapolation_value = 0;
    let _keep_aspect_ratio_policy = "stretch";
    let _mode = "nearest";
    let _nearest_mode = "floor";
    todo!()
}
