use crate::tensor::Tensor;
use anyhow::{Result, anyhow};

pub fn conv(input: &Tensor, weight: &Tensor, bias: &Tensor, name: &str) -> Result<Tensor> {
    // ESRGAN hardcoded attributes
    let _auto_pad = "NOTSET";
    let _group = 1;
    let _dilations = [1, 1];
    let _pads = [1, 1, 1, 1];
    let _strides = [1, 1];

    let in_dims = input.dims();
    let w_dims = weight.dims();

    if in_dims.len() != 4 || w_dims.len() != 4 {
        return Err(anyhow!("conv: input and weight must be 4D tensors"));
    }

    let [n, c_in, h, w] = [
        in_dims[0] as usize,
        in_dims[1] as usize,
        in_dims[2] as usize,
        in_dims[3] as usize,
    ];
    let [c_out, w_c_in, kh, kw] = [
        w_dims[0] as usize,
        w_dims[1] as usize,
        w_dims[2] as usize,
        w_dims[3] as usize,
    ];

    if c_in != w_c_in {
        return Err(anyhow::anyhow!(
            "{name}: input channels {} != weight in_channels {}",
            c_in,
            w_c_in
        ));
    }
    if kh != 3 || kw != 3 {
        return Err(anyhow::anyhow!(
            "{name}: expected 3x3 kernel, got {}x{}",
            kh,
            kw
        ));
    }

    let bias_data = bias.data();
    if bias_data.len() != c_out {
        return Err(anyhow::anyhow!(
            "{name}: bias len {} != out_channels {}",
            bias_data.len(),
            c_out
        ));
    }

    let in_data = input.data();
    let w_data = weight.data();

    let out_h = h;
    let out_w = w;
    let mut output = vec![0.0f32; n * c_out * out_h * out_w];

    for n_idx in 0..n {
        for oc in 0..c_out {
            let mut acc = bias_data[oc];

            // weight base for this output channel: [oc, :, :, :]
            let w_base = oc * c_in * 3 * 3;

            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = acc;

                    // unroll the 3x3 kernel since it's fixed size.
                    // kh, kw in {0,1,2}; input offset = oh+kh-1, ow+kw-1
                    for kh in 0..3 {
                        let ih = oh as i32 + kh as i32 - 1;
                        if ih < 0 || ih >= h as i32 {
                            continue;
                        }
                        let ih = ih as usize;

                        for kw in 0..3 {
                            let iw = ow as i32 + kw as i32 - 1;
                            if iw < 0 || iw >= w as i32 {
                                continue;
                            }
                            let iw = iw as usize;

                            // input[n, ic, ih, iw]
                            let input_base = (n_idx * c_in + 0) * h * w;
                            let input_row = ih * w;

                            // weight[oc, ic, kh, kw]
                            let w_row = w_base + kh * 3 + kw;

                            for ic in 0..c_in {
                                let input_idx = input_base + ic * h * w + input_row + iw;
                                let weight_idx = w_row + ic * 3 * 3;
                                sum += in_data[input_idx] * w_data[weight_idx];
                            }
                        }
                    }

                    let out_idx = ((n_idx * c_out + oc) * out_h + oh) * out_w + ow;
                    output[out_idx] = sum;
                    acc = 0.0;
                }
            }
        }
    }

    let dims = vec![n as i64, c_out as i64, out_h as i64, out_w as i64];
    Tensor::create(name, dims, output)
}

pub fn leaky_relu(input: &Tensor, name: &str) -> Result<Tensor> {
    let alpha = 0.2; // ESRGAN hardcoded attributes
    let data = input.data();
    let output_data: Vec<f32> = data
        .iter()
        .map(|&x| if x < 0.0 { x * alpha } else { x })
        .collect();
    Tensor::create(name, input.dims().to_vec(), output_data)
}

pub fn concat() {}
pub fn add() {}
pub fn mul() {}
pub fn resize() {}
