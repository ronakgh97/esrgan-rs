use crate::tensor::Tensor;
use anyhow::Result;
use anyhow::anyhow;

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

    #[cfg(not(target_arch = "x86_64"))]
    fn leaky_relu_inner(input: &[f32], alpha: f32) -> Result<Vec<f32>> {
        let output_data: Vec<f32> = input
            .iter()
            .map(|&x| if x < 0.0 { x * alpha } else { x })
            .collect();
        Ok(output_data)
    }
    #[cfg(target_arch = "x86_64")]
    fn leaky_relu_inner(input: &[f32], alpha: f32) -> Result<Vec<f32>> {
        let output_data: Vec<f32> = input
            .iter()
            .map(|&x| if x < 0.0 { x * alpha } else { x })
            .collect();
        Ok(output_data)
    }
    
    let output_data = leaky_relu_inner(data, ALPHA)?;
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

fn broadcast_op<OP>(x: &Tensor, y: &Tensor, output: &str, scalar_op: OP) -> Result<Tensor>
where
    OP: Fn(f32, f32) -> f32,
{
    let x_dims = x.dims();
    let y_dims = y.dims();
    let x_strides = x.strides();
    let y_strides = y.strides();

    let rank = x_dims.len().max(y_dims.len());
    let x_pad = rank - x_dims.len();
    let y_pad = rank - y_dims.len();

    let mut out_dims = Vec::with_capacity(rank);
    let mut bx_strides = vec![0usize; rank];
    let mut by_strides = vec![0usize; rank];

    for k in 0..rank {
        let xd = if k < x_pad { 1 } else { x_dims[k - x_pad] };
        let yd = if k < y_pad { 1 } else { y_dims[k - y_pad] };

        let od = if xd == yd {
            xd
        } else if xd == 1 {
            yd
        } else if yd == 1 {
            xd
        } else {
            return Err(anyhow!(
                "{}: cannot broadcast X dims {:?} with Y dims {:?}",
                output,
                x_dims,
                y_dims
            ));
        };

        if k >= x_pad && xd != 1 {
            bx_strides[k] = x_strides[k - x_pad];
        }
        if k >= y_pad && yd != 1 {
            by_strides[k] = y_strides[k - y_pad];
        }

        out_dims.push(od);
    }

    let total: usize = out_dims
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d as usize))
        .ok_or_else(|| anyhow!("{}: dims overflow", output))?;

    for k in 0..rank {
        if out_dims[k] == 1 {
            bx_strides[k] = 0;
            by_strides[k] = 0;
        }
    }

    let mut axes: Vec<(usize, usize, usize)> = Vec::with_capacity(rank);
    for k in (0..rank).rev() {
        let d = out_dims[k] as usize;
        let xb = bx_strides[k];
        let yb = by_strides[k];

        let merged = match axes.last_mut() {
            Some(last) => {
                let (ld, lx, ly) = *last;
                let x_ok = (xb == 0 && lx == 0) || (xb != 0 && lx != 0 && xb == lx * ld);
                let y_ok = (yb == 0 && ly == 0) || (yb != 0 && ly != 0 && yb == ly * ld);
                if x_ok && y_ok {
                    *last = (ld * d, lx, ly);
                    true
                } else {
                    false
                }
            }
            None => false,
        };
        if !merged {
            axes.push((d, xb, yb));
        }
    }
    axes.reverse();

    let d_rank = axes.len();
    let mut data = Vec::with_capacity(total);
    let data_ptr: *mut f32 = data.as_mut_ptr();

    let xp = x.data().as_ptr();
    let yp = y.data().as_ptr();

    let mut coords = vec![0usize; d_rank];
    let (mut xi, mut yi) = (0usize, 0usize);

    unsafe {
        for idx in 0..total {
            *data_ptr.add(idx) = scalar_op(*xp.add(xi), *yp.add(yi));
            for k in (0..d_rank).rev() {
                coords[k] += 1;
                let (d, bx, by) = axes[k];
                if coords[k] < d {
                    xi += bx;
                    yi += by;
                    break;
                }
                coords[k] = 0;
                xi -= (d - 1) * bx;
                yi -= (d - 1) * by;
            }
        }
        data.set_len(total);
    }

    Tensor::create(output, out_dims.to_vec(), data)
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
