use crate::tensor::Tensor;
use anyhow::Result;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

pub fn conv(input: &Tensor, weight: &Tensor, bias: &Tensor, name: &str) -> Result<Tensor> {
    // ESRGAN hardcoded attributes
    // let auto_pad = "NOTSET";
    // let group = 1;
    // let dilations = [1, 1];
    // let pads = [1, 1, 1, 1];
    // let strides = [1, 1];
    todo!()
}

pub fn leaky_relu(input: &Tensor, name: &str) -> Result<Tensor> {
    // ESRGAN hardcoded attributes
    let alpha = 0.20000000298023224;
    let data = &input.data;
    let dim = &input.dims;

    #[cfg(not(target_arch = "x86_64"))]
    fn leaky_relu_inner(input: &[f32], alpha: f32) -> Result<Vec<f32>> {
        let output_data: Vec<f32> = input
            .iter()
            .map(|&x| if x < 0.0 { x.algebraic_mul(alpha) } else { x })
            .collect();
        Ok(output_data)
    }
    #[cfg(target_arch = "x86_64")]
    fn leaky_relu_inner(input: &[f32], alpha: f32) -> Result<Vec<f32>> {
        let mut data = vec![0f32; input.len()];

        if data.len() >= 8 {
            unsafe {
                let zero = _mm256_setzero_ps();
                let va = _mm256_set1_ps(alpha);

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
                    *out = if v < 0.0 { alpha * v } else { v };
                }
            }
        } else {
            for (&v, out) in input.iter().zip(data.iter_mut()) {
                *out = if v < 0.0 { alpha * v } else { v };
            }
        }
        Ok(data)
    }

    let output_data = leaky_relu_inner(data, alpha)?;
    Tensor::create(name, dim.to_vec(), output_data)
}

pub fn concat(input_a: &Tensor, input_b: &Tensor, name: &str) -> Result<Tensor> {
    // ESRGAN hardcoded attributes
    // let axis = 1;
    let a_dims = &input_a.dims;
    let b_dims = &input_b.dims;

    assert_eq!(
        a_dims.len(),
        4,
        "ESRGAN's concat operation requires rank-4 tensors"
    );
    assert_eq!(
        b_dims.len(),
        4,
        "ESRGAN's concat operation requires rank-4 tensors"
    );
    assert!(
        a_dims[0] == b_dims[0] && a_dims[2] == b_dims[2] && a_dims[3] == b_dims[3],
        "ESRGAN's concat operation requires equal N, H, W, got {a_dims:?} and {b_dims:?}"
    );

    let n = a_dims[0] as usize;
    let ca = a_dims[1] as usize;
    let cb = b_dims[1] as usize;
    let hw = (a_dims[2] * a_dims[3]) as usize;

    // In NCHW every per-batch slab is contiguous, so an axis-1 concat is just
    // pasting b's slab right after a's slab per batch (2 memcpys per batch)
    let a_data = &input_a.data;
    let b_data = &input_b.data;
    let mut data = Vec::with_capacity(n * (ca + cb) * hw);
    for i in 0..n {
        data.extend_from_slice(&a_data[i * ca * hw..(i + 1) * ca * hw]);
        data.extend_from_slice(&b_data[i * cb * hw..(i + 1) * cb * hw]);
    }

    Tensor::create(
        name,
        vec![a_dims[0], a_dims[1] + b_dims[1], a_dims[2], a_dims[3]],
        data,
    )
}

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

pub fn mul(input_a: &Tensor, input_b: &Tensor, name: &str) -> Result<Tensor> {
    // ESRGAN's all mul operation are tensor * rank-0 aka scalar
    let alpha = 0.20000000298023224;
    let a_dim = &input_a.dims;
    let n = input_a.data.len();

    assert!(
        input_b.dims.is_empty(),
        "ESRGAN's mul operation requires rank-0 tensor as second input"
    );
    assert_eq!(
        input_b.data.as_ref(),
        &[alpha],
        "ESRGAN's mul operation requires second input to be rank-0 tensor with value 0.20000000298023224"
    );

    #[cfg(not(target_arch = "x86_64"))]
    let data: Vec<f32> = input_a
        .data()
        .iter()
        .map(|a| a.algebraic_mul(alpha))
        .collect();

    #[cfg(target_arch = "x86_64")]
    let data = {
        let mut data = Vec::with_capacity(n);
        let dp: *mut f32 = data.as_mut_ptr();
        let ap = input_a.data.as_ptr();

        unsafe {
            let mut i = 0;
            let alpha_v = _mm256_set1_ps(alpha);
            while i + 32 <= n {
                let a0 = _mm256_loadu_ps(ap.add(i));
                let a1 = _mm256_loadu_ps(ap.add(i + 8));
                let a2 = _mm256_loadu_ps(ap.add(i + 16));
                let a3 = _mm256_loadu_ps(ap.add(i + 24));
                _mm256_storeu_ps(dp.add(i), _mm256_mul_ps(a0, alpha_v));
                _mm256_storeu_ps(dp.add(i + 8), _mm256_mul_ps(a1, alpha_v));
                _mm256_storeu_ps(dp.add(i + 16), _mm256_mul_ps(a2, alpha_v));
                _mm256_storeu_ps(dp.add(i + 24), _mm256_mul_ps(a3, alpha_v));
                i += 32;
            }
            while i + 8 <= n {
                let a = _mm256_loadu_ps(ap.add(i));
                _mm256_storeu_ps(dp.add(i), _mm256_mul_ps(a, alpha_v));
                i += 8;
            }
            while i < n {
                *dp.add(i) = *ap.add(i) * alpha;
                i += 1;
            }
            data.set_len(n);
        }
        data
    };

    Tensor::create(name, a_dim.to_vec(), data)
}

pub fn resize(input: &Tensor, name: &str) -> Result<Tensor> {
    // ESRGAN hardcoded attributes
    // let _antialias = 0;
    // let _coordinate_transformation_mode = "asymmetric";
    // let _cubic_coeff_a = -0.75;
    // let _exclude_outside = 0;
    // let _extrapolation_value = 0;
    // let _keep_aspect_ratio_policy = "stretch";
    // let _mode = "nearest";
    // let _nearest_mode = "floor";

    // scales = [1, 1, 2, 2]: both `upsample_nearest2d` nodes upscale x2 (x4 total for the model)
    // nearest + floor + asymmetric => out(o) = in(o / 2) with integer division
    let dims = &input.dims;

    assert_eq!(
        dims.len(),
        4,
        "ESRGAN's resize operation requires rank-4 tensor"
    );

    let [n, c, h, w] = [
        dims[0] as usize,
        dims[1] as usize,
        dims[2] as usize,
        dims[3] as usize,
    ];
    let out_w = w * 2;

    #[cfg(not(target_arch = "x86_64"))]
    fn resize_nearest_x2_row(input: &[f32], output: &mut [f32]) {
        for (x, &v) in input.iter().enumerate() {
            output[2 * x] = v;
            output[2 * x + 1] = v;
        }
    }

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

    // Each input row expands to a 2W row, which is pasted into output rows 2y and 2y+1
    // The expanded row stays L1-resident between the two pastes, so
    // DRAM traffic is exactly: input read once, output written once
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
