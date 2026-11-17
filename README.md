**esrgan-rs** - running OG ESRGAN using pure rust on CPU (AVX2 accelerated + fallback) and no python, no torch or cuda.

The [model](https://aihub.qualcomm.com/compute/models/esrgan) used is from Qualcomm AI Hub as .onnx format.

```text
esrgan-onnx-float/
    esrgan.data
    esrgan.onnx
    esrgan_single.onnx
    merge.py # for merging data and graph
    metadata.json
```

The model is provided as `.onnx` and `.data` files, but in order to collect the weights and parse the tensor names
together initially, It's needs to be merged using `merge.py`. So it can read as single `.onnx` file,
which can also be used in other runtimes.

The project contains only three files (690 loc):

- [lib](src/lib.rs) - main library file (init_weights and run)
- [kernels](src/kernels.rs) - kernel implementations (conv, leaky relu, concat, add, mul and resize)
- [tensor](src/tensor.rs) - basic tensor utilities

> NOTE: Other models or similar variants from different sources from the internet may fail, as the code is hardcoded.

---

Checkout my other [project](https://github.com/ronakgh97/onnx_rs) about minimal onnx-runtime in pure rust