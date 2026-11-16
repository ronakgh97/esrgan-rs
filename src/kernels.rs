use crate::tensor::Tensor;
use anyhow::Result;
use anyhow::anyhow;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

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
        let mut data = vec![0f32; input.len()];

        if data.len() >= 8 {
            // simd path
            unsafe {
                let zero = _mm256_setzero_ps();
                let va = _mm256_set1_ps(alpha);

                // unroll by 4 ymm per iteration, same skeleton as `relu`
                let (input_chunk32, input_rem32) = input.as_chunks::<32>();
                let (output_chunk32, output_rem32) = data.as_chunks_mut::<32>();

                for (input, output) in input_chunk32.iter().zip(output_chunk32.iter_mut()) {
                    let v0 = _mm256_loadu_ps(input.as_ptr());
                    let v1 = _mm256_loadu_ps(input.as_ptr().add(8));
                    let v2 = _mm256_loadu_ps(input.as_ptr().add(16));
                    let v3 = _mm256_loadu_ps(input.as_ptr().add(24));

                    let a0 = _mm256_mul_ps(v0, va);
                    let a1 = _mm256_mul_ps(v1, va);
                    let a2 = _mm256_mul_ps(v2, va);
                    let a3 = _mm256_mul_ps(v3, va);

                    let m0 = _mm256_cmp_ps(v0, zero, _CMP_LT_OQ);
                    let m1 = _mm256_cmp_ps(v1, zero, _CMP_LT_OQ);
                    let m2 = _mm256_cmp_ps(v2, zero, _CMP_LT_OQ);
                    let m3 = _mm256_cmp_ps(v3, zero, _CMP_LT_OQ);

                    let v0 = _mm256_blendv_ps(v0, a0, m0);
                    let v1 = _mm256_blendv_ps(v1, a1, m1);
                    let v2 = _mm256_blendv_ps(v2, a2, m2);
                    let v3 = _mm256_blendv_ps(v3, a3, m3);

                    _mm256_storeu_ps(output.as_mut_ptr(), v0);
                    _mm256_storeu_ps(output.as_mut_ptr().add(8), v1);
                    _mm256_storeu_ps(output.as_mut_ptr().add(16), v2);
                    _mm256_storeu_ps(output.as_mut_ptr().add(24), v3);
                }

                // simd tail (< 32 floats)
                let (input_chunk8, input_rem8) = input_rem32.as_chunks::<8>();
                let (output_chunk8, output_rem8) = output_rem32.as_chunks_mut::<8>();
                for (input, output) in input_chunk8.iter().zip(output_chunk8.iter_mut()) {
                    let v = _mm256_loadu_ps(input.as_ptr());
                    let a = _mm256_mul_ps(v, va);
                    let m = _mm256_cmp_ps(v, zero, _CMP_LT_OQ);
                    _mm256_storeu_ps(output.as_mut_ptr(), _mm256_blendv_ps(v, a, m));
                }

                // scalar (< 8 floats)
                for (&v, out) in input_rem8.iter().zip(output_rem8.iter_mut()) {
                    *out = if v < 0.0 { alpha * v } else { v };
                }
            }
        } else {
            // scalar path
            for (&v, out) in input.iter().zip(data.iter_mut()) {
                *out = if v < 0.0 { alpha * v } else { v };
            }
        }
        Ok(data)
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
    let a_dim = input_a.dims();
    let b_dim = input_b.dims();
    let n = input_a.data().len();

    #[cfg(not(target_arch = "x86_64"))]
    if a_dim == b_dim {
        let data: Vec<f32> = input_a
            .data()
            .iter()
            .zip(input_b.data().iter())
            .map(|(a, b)| a + b)
            .collect();
        return Tensor::create(name, a_dim.to_vec(), data);
    }

    #[cfg(target_arch = "x86_64")]
    if a_dim == b_dim {
        let mut data = Vec::with_capacity(n);
        let dp: *mut f32 = data.as_mut_ptr();
        let ap = input_a.data().as_ptr();
        let bp = input_b.data().as_ptr();

        unsafe {
            let mut i = 0usize;
            while i + 32 <= n {
                let a0 = _mm256_loadu_ps(ap.add(i));
                let a1 = _mm256_loadu_ps(ap.add(i + 8));
                let a2 = _mm256_loadu_ps(ap.add(i + 16));
                let a3 = _mm256_loadu_ps(ap.add(i + 24));
                let b0 = _mm256_loadu_ps(bp.add(i));
                let b1 = _mm256_loadu_ps(bp.add(i + 8));
                let b2 = _mm256_loadu_ps(bp.add(i + 16));
                let b3 = _mm256_loadu_ps(bp.add(i + 24));
                _mm256_storeu_ps(dp.add(i), _mm256_sub_ps(a0, b0));
                _mm256_storeu_ps(dp.add(i + 8), _mm256_sub_ps(a1, b1));
                _mm256_storeu_ps(dp.add(i + 16), _mm256_sub_ps(a2, b2));
                _mm256_storeu_ps(dp.add(i + 24), _mm256_sub_ps(a3, b3));
                i += 32;
            }
            while i + 8 <= n {
                let a = _mm256_loadu_ps(ap.add(i));
                let b = _mm256_loadu_ps(bp.add(i));
                _mm256_storeu_ps(dp.add(i), _mm256_sub_ps(a, b));
                i += 8;
            }
            while i < n {
                *dp.add(i) = *ap.add(i) - *bp.add(i);
                i += 1;
            }
            data.set_len(n);
        }
        return Tensor::create(name, a_dim.to_vec(), data);
    }

    broadcast_op(input_a, input_b, name, |a, b| a.algebraic_add(b))
}
pub fn mul(input_a: &Tensor, input_b: &Tensor, name: &str) -> Result<Tensor> {
    let a_dim = input_a.dims().len();
    let b_dim = input_b.dims().len();

    if a_dim == b_dim {
        todo!()
    }
    broadcast_op(input_a, input_b, name, |a, b| a.algebraic_mul(b))
}

#[inline(always)]
fn broadcast_op<OP>(x: &Tensor, y: &Tensor, name: &str, scalar_op: OP) -> Result<Tensor>
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
                "tensor '{}': cannot broadcast X dims {:?} with Y dims {:?}",
                name,
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

    let total: usize = out_dims.iter().map(|d| *d as usize).product();
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

    Tensor::create(name, out_dims.to_vec(), data)
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
