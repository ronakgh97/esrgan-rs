use crate::tensor::Tensor;
use anyhow::Result;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline]
pub fn conv(input: &Tensor, weight: &Tensor, bias: &Tensor, name: &str) -> Result<Tensor> {
    // ESRGAN hardcoded attributes
    // let auto_pad = "NOTSET";
    // let group = 1;
    // let dilations = [1, 1];
    // let pads = [1, 1, 1, 1];
    // let strides = [1, 1];
    let in_dims = &input.dims;
    let w_dims = &weight.dims;
    let b_dims = &bias.dims;

    assert_eq!(
        in_dims.len(),
        4,
        "ESRGAN's conv operation requires rank-4 input"
    );
    assert_eq!(
        w_dims.len(),
        4,
        "ESRGAN's conv operation requires rank-4 weight"
    );
    assert_eq!(
        (&w_dims[2], &w_dims[3]),
        (&3, &3),
        "ESRGAN's conv operation requires 3x3 kernels"
    );
    assert_eq!(
        w_dims[1], in_dims[1],
        "ESRGAN's conv operation requires weight Cin to match input channels"
    );
    assert_eq!(
        b_dims.len(),
        1,
        "ESRGAN's conv operation requires rank-1 bias"
    );
    assert_eq!(
        b_dims[0], w_dims[0],
        "ESRGAN's conv operation requires bias length to match Cout"
    );

    #[inline(always)]
    #[cfg(not(target_arch = "x86_64"))]
    fn row_axpy(out_row: &mut [f32], in_row: &[f32], k: f32) {
        for (o, &i) in out_row.iter_mut().zip(in_row.iter()) {
            *o += k.algebraic_mul(i);
        }
    }
    #[inline(always)]
    #[cfg(target_arch = "x86_64")]
    fn row_axpy(out_row: &mut [f32], in_row: &[f32], k: f32) {
        let n = out_row.len();
        assert_eq!(
            n,
            in_row.len(),
            "row_axpy: out_row and in_row must have the same length"
        );

        unsafe {
            let op = out_row.as_mut_ptr();
            let ip = in_row.as_ptr();
            let vk = _mm256_set1_ps(k);

            let mut x = 0usize;
            while x + 32 <= n {
                let o0 = _mm256_loadu_ps(op.add(x));
                let o1 = _mm256_loadu_ps(op.add(x + 8));
                let o2 = _mm256_loadu_ps(op.add(x + 16));
                let o3 = _mm256_loadu_ps(op.add(x + 24));
                let i0 = _mm256_loadu_ps(ip.add(x));
                let i1 = _mm256_loadu_ps(ip.add(x + 8));
                let i2 = _mm256_loadu_ps(ip.add(x + 16));
                let i3 = _mm256_loadu_ps(ip.add(x + 24));
                _mm256_storeu_ps(op.add(x), _mm256_fmadd_ps(vk, i0, o0));
                _mm256_storeu_ps(op.add(x + 8), _mm256_fmadd_ps(vk, i1, o1));
                _mm256_storeu_ps(op.add(x + 16), _mm256_fmadd_ps(vk, i2, o2));
                _mm256_storeu_ps(op.add(x + 24), _mm256_fmadd_ps(vk, i3, o3));
                x += 32;
            }
            while x + 8 <= n {
                let o = _mm256_loadu_ps(op.add(x));
                let i = _mm256_loadu_ps(ip.add(x));
                _mm256_storeu_ps(op.add(x), _mm256_fmadd_ps(vk, i, o));
                x += 8;
            }
            while x < n {
                *op.add(x) += k * *ip.add(x);
                x += 1;
            }
        }
    }

    let n = in_dims[0] as usize;
    let cin = in_dims[1] as usize;
    let h = in_dims[2] as usize;
    let w = in_dims[3] as usize;
    let cout = w_dims[0] as usize;

    let in_data = &input.data;
    let weight_data = &weight.data;
    let bias_data = &bias.data;

    // zero-padding into [n, cin, h+2, w+2] once, so every kernel tap is always
    // in-bounds and the hot loop stays branch-free
    let pw = w + 2;
    let ph = h + 2;
    let mut padded = vec![0f32; n * cin * ph * pw];
    for plane in 0..n * cin {
        let src = &in_data[plane * h * w..(plane + 1) * h * w];
        let dst = &mut padded[plane * ph * pw..(plane + 1) * ph * pw];
        for y in 0..h {
            dst[(y + 1) * pw + 1..(y + 1) * pw + 1 + w].copy_from_slice(&src[y * w..(y + 1) * w]);
        }
    }

    // parallelize over horizontal bands, so every task computes ALL output channels
    // for its rows, reading its input band from the padded buffer,
    // each input byte is read from DRAM exactly once
    let threads = rayon::current_num_threads().max(1);
    let mut bands = threads.min(h).max(1);
    while !h.is_multiple_of(bands) {
        bands -= 1;
    }
    let r = h / bands;

    let mut data = vec![0f32; n * cout * h * w];
    data.par_chunks_mut(r * w)
        .enumerate()
        .for_each(|(idx, out_band)| {
            let n_i = idx / (cout * bands);
            let co = (idx / bands) % cout;
            let band = idx % bands;
            let wbase_co = co * cin * 9;
            let bias_v = bias_data[co];

            for y in 0..r {
                let oy = band * r + y;
                let out_row = &mut out_band[y * w..(y + 1) * w];
                out_row.fill(bias_v);
                for ci in 0..cin {
                    let iplane = (n_i * cin + ci) * ph * pw;
                    let wtaps = &weight_data[wbase_co + ci * 9..wbase_co + ci * 9 + 9];
                    for kh in 0..3 {
                        let irow = iplane + (oy + kh) * pw;
                        for kw in 0..3 {
                            row_axpy(
                                out_row,
                                &padded[irow + kw..irow + kw + w],
                                wtaps[kh * 3 + kw],
                            );
                        }
                    }
                }
            }
        });

    Tensor::create(
        name,
        vec![in_dims[0], w_dims[0], in_dims[2], in_dims[3]],
        data,
    )
}

#[inline]
pub fn leaky_relu(input: &Tensor, alpha: f32, name: &str) -> Result<Tensor> {
    // ESRGAN hardcoded attributes
    #[allow(clippy::excessive_precision)]
    const ALPHA: f32 = 0.20000000298023224;
    let data = &input.data;
    let dim = &input.dims;

    assert_eq!(
        alpha, ALPHA,
        "ESRGAN's leaky_relu operation requires alpha=0.2"
    );

    #[inline(always)]
    #[cfg(not(target_arch = "x86_64"))]
    fn leaky_relu_inner(input: &[f32]) -> Result<Vec<f32>> {
        let output_data: Vec<f32> = input
            .iter()
            .map(|&x| if x < 0.0 { x.algebraic_mul(ALPHA) } else { x })
            .collect();
        Ok(output_data)
    }
    #[inline(always)]
    #[cfg(target_arch = "x86_64")]
    fn leaky_relu_inner(input: &[f32]) -> Result<Vec<f32>> {
        let mut data = vec![0f32; input.len()];

        if data.len() >= 8 {
            unsafe {
                let zero = _mm256_setzero_ps();
                let va = _mm256_set1_ps(ALPHA);

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
                let (input_chunk8, input_rem8) = input_rem32.as_chunks::<8>();
                let (output_chunk8, output_rem8) = output_rem32.as_chunks_mut::<8>();
                for (input, output) in input_chunk8.iter().zip(output_chunk8.iter_mut()) {
                    let v = _mm256_loadu_ps(input.as_ptr());
                    let a = _mm256_mul_ps(v, va);
                    let m = _mm256_cmp_ps(v, zero, _CMP_LT_OQ);
                    _mm256_storeu_ps(output.as_mut_ptr(), _mm256_blendv_ps(v, a, m));
                }
                for (&v, out) in input_rem8.iter().zip(output_rem8.iter_mut()) {
                    *out = if v < 0.0 { ALPHA.algebraic_mul(v) } else { v };
                }
            }
        } else {
            for (&v, out) in input.iter().zip(data.iter_mut()) {
                *out = if v < 0.0 { ALPHA.algebraic_mul(v) } else { v };
            }
        }
        Ok(data)
    }

    let output_data = leaky_relu_inner(data)?;
    Tensor::create(name, dim.to_vec(), output_data)
}

#[inline]
pub fn concat(inputs: &[Tensor], name: &str) -> Result<Tensor> {
    // ESRGAN hardcoded attributes
    // let axis = 1;
    assert!(
        !inputs.is_empty(),
        "ESRGAN's concat operation requires at least one input"
    );
    let first_dims = &inputs[0].dims;
    for t in inputs {
        let d = &t.dims;
        assert_eq!(
            d.len(),
            4,
            "ESRGAN's concat operation requires rank-4 tensors"
        );
        assert!(
            d[0] == first_dims[0] && d[2] == first_dims[2] && d[3] == first_dims[3],
            "ESRGAN's concat operation requires equal N, H, W, got {d:?} and {first_dims:?}"
        );
    }

    let n = first_dims[0] as usize;
    let hw = (first_dims[2] * first_dims[3]) as usize;
    let cout = inputs.iter().map(|t| t.dims[1]).sum::<i64>();

    // since layout is NCHW, every per-batch slab is contiguous, so an axis-1 concat is just
    // pasting every input's slab in order per batch (1 memcpy per input)
    let mut data = Vec::with_capacity(n * cout as usize * hw);
    for i in 0..n {
        for t in inputs {
            let c = t.dims[1] as usize;
            data.extend_from_slice(&t.data[i * c * hw..(i + 1) * c * hw]);
        }
    }

    Tensor::create(
        name,
        vec![first_dims[0], cout, first_dims[2], first_dims[3]],
        data,
    )
}

#[inline]
pub fn add(input_a: &Tensor, input_b: &Tensor, name: &str) -> Result<Tensor> {
    let a_dim = &input_a.dims;
    let b_dim = &input_b.dims;
    let n = input_a.data.len();

    assert_eq!(
        a_dim, b_dim,
        "ESRGAN's add operation requires inputs with the same shape"
    );

    #[cfg(not(target_arch = "x86_64"))]
    let data: Vec<f32> = input_a
        .data()
        .iter()
        .zip(input_b.data().iter())
        .map(|(a, b)| a.algebraic_add(b))
        .collect();

    #[cfg(target_arch = "x86_64")]
    let data = {
        let mut data = Vec::with_capacity(n);
        let dp: *mut f32 = data.as_mut_ptr();
        let ap = input_a.data.as_ptr();
        let bp = input_b.data.as_ptr();

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
                _mm256_storeu_ps(dp.add(i), _mm256_add_ps(a0, b0));
                _mm256_storeu_ps(dp.add(i + 8), _mm256_add_ps(a1, b1));
                _mm256_storeu_ps(dp.add(i + 16), _mm256_add_ps(a2, b2));
                _mm256_storeu_ps(dp.add(i + 24), _mm256_add_ps(a3, b3));
                i += 32;
            }
            while i + 8 <= n {
                let a = _mm256_loadu_ps(ap.add(i));
                let b = _mm256_loadu_ps(bp.add(i));
                _mm256_storeu_ps(dp.add(i), _mm256_add_ps(a, b));
                i += 8;
            }
            while i < n {
                *dp.add(i) = *ap.add(i) + *bp.add(i);
                i += 1;
            }
            data.set_len(n);
        }
        data
    };

    Tensor::create(name, a_dim.to_vec(), data)

    // ESRGAN's all add operations does not require broadcasting, they have same shape
    // broadcast_op(input_a, input_b, name, |a, b| a.algebraic_add(b))
}

#[inline]
pub fn mul(input_a: &Tensor, input_b: &Tensor, name: &str) -> Result<Tensor> {
    // ESRGAN's all mul operation are tensor * rank-0 aka scalar
    #[allow(clippy::excessive_precision)]
    const SCALAR: f32 = 0.20000000298023224;
    let a_dim = &input_a.dims;
    let n = input_a.data.len();

    assert!(
        input_b.dims.is_empty(),
        "ESRGAN's mul operation requires rank-0 tensor as second input"
    );
    assert_eq!(
        input_b.data.as_ref(),
        &[SCALAR],
        "ESRGAN's mul operation requires second input to be rank-0 tensor with value 0.20000000298023224"
    );

    #[cfg(not(target_arch = "x86_64"))]
    let data: Vec<f32> = input_a
        .data()
        .iter()
        .map(|a| a.algebraic_mul(SCALAR))
        .collect();

    #[cfg(target_arch = "x86_64")]
    let data = {
        let mut data = Vec::with_capacity(n);
        let dp: *mut f32 = data.as_mut_ptr();
        let ap = input_a.data.as_ptr();

        unsafe {
            let mut i = 0;
            let scalar_v = _mm256_set1_ps(SCALAR);
            while i + 32 <= n {
                let a0 = _mm256_loadu_ps(ap.add(i));
                let a1 = _mm256_loadu_ps(ap.add(i + 8));
                let a2 = _mm256_loadu_ps(ap.add(i + 16));
                let a3 = _mm256_loadu_ps(ap.add(i + 24));
                _mm256_storeu_ps(dp.add(i), _mm256_mul_ps(a0, scalar_v));
                _mm256_storeu_ps(dp.add(i + 8), _mm256_mul_ps(a1, scalar_v));
                _mm256_storeu_ps(dp.add(i + 16), _mm256_mul_ps(a2, scalar_v));
                _mm256_storeu_ps(dp.add(i + 24), _mm256_mul_ps(a3, scalar_v));
                i += 32;
            }
            while i + 8 <= n {
                let a = _mm256_loadu_ps(ap.add(i));
                _mm256_storeu_ps(dp.add(i), _mm256_mul_ps(a, scalar_v));
                i += 8;
            }
            while i < n {
                *dp.add(i) = *ap.add(i) * SCALAR;
                i += 1;
            }
            data.set_len(n);
        }
        data
    };

    Tensor::create(name, a_dim.to_vec(), data)
}

#[inline]
pub fn resize(input: &Tensor, scale: &Tensor, name: &str) -> Result<Tensor> {
    // ESRGAN hardcoded attributes
    // let _antialias = 0;
    // let _coordinate_transformation_mode = "asymmetric";
    // let _cubic_coeff_a = -0.75;
    // let _exclude_outside = 0;
    // let _extrapolation_value = 0;
    // let _keep_aspect_ratio_policy = "stretch";
    // let _mode = "nearest";
    // let _nearest_mode = "floor";

    let dims = &input.dims;
    assert_eq!(
        dims.len(),
        4,
        "ESRGAN's resize operation requires rank-4 tensor"
    );

    // scales = [1, 1, 2, 2]: both `upsample_nearest2d` nodes upscale x2 (x4 total for the model)
    // nearest + floor + asymmetric => out(o) = in(o / 2) with integer division
    let scale_data = &scale.data;
    assert_eq!(
        scale_data.as_ref(),
        &[1.0, 1.0, 2.0, 2.0],
        "ESRGAN's resize operation requires rank-1 tensor with value [1, 1, 2, 2]"
    );

    let [n, c, h, w] = [
        dims[0] as usize,
        dims[1] as usize,
        dims[2] as usize,
        dims[3] as usize,
    ];
    let out_w = w * 2;

    #[inline(always)]
    #[cfg(not(target_arch = "x86_64"))]
    fn resize_nearest_x2_row(input: &[f32], output: &mut [f32]) {
        for (x, &v) in input.iter().enumerate() {
            output[2 * x] = v;
            output[2 * x + 1] = v;
        }
    }
    #[inline(always)]
    #[cfg(target_arch = "x86_64")]
    fn resize_nearest_x2_row(input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(output.len(), input.len() * 2);

        let (input_chunk8, input_rem8) = input.as_chunks::<8>();
        let (output_chunk16, _) = output.as_chunks_mut::<16>();
        let dup_low = unsafe { _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3) };
        let dup_high = unsafe { _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7) };

        for (i8, o16) in input_chunk8.iter().zip(output_chunk16.iter_mut()) {
            unsafe {
                let v = _mm256_loadu_ps(i8.as_ptr());
                // [v0 v0 v1 v1 v2 v2 v3 v3] and [v4 v4 v5 v5 v6 v6 v7 v7]
                let lo = _mm256_permutevar8x32_ps(v, dup_low);
                let hi = _mm256_permutevar8x32_ps(v, dup_high);
                _mm256_storeu_ps(o16.as_mut_ptr(), lo);
                _mm256_storeu_ps(o16.as_mut_ptr().add(8), hi);
            }
        }

        let tail = input_chunk8.len() * 16;
        for (x, &v) in input_rem8.iter().enumerate() {
            output[tail + 2 * x] = v;
            output[tail + 2 * x + 1] = v;
        }
    }

    // each input row expands to a 2W row, which is pasted into output rows 2y and 2y+1,
    // the expanded row stays L1-resident between the two pastes, so
    // DRAM traffic is exactly, input read once and output written once
    let input_data = &input.data;
    let mut data = Vec::with_capacity(n * c * h * 4 * w);
    let mut row_buf = vec![0f32; out_w];
    for plane in 0..n * c {
        let in_plane = &input_data[plane * h * w..(plane + 1) * h * w];
        for in_row in in_plane.chunks_exact(w) {
            resize_nearest_x2_row(in_row, &mut row_buf);
            data.extend_from_slice(&row_buf);
            data.extend_from_slice(&row_buf);
        }
    }

    Tensor::create(name, vec![dims[0], dims[1], 2 * dims[2], 2 * dims[3]], data)
}
