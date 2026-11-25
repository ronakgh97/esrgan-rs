**esrgan-rs** - running OG ESRGAN using pure rust on CPU (AVX2 accelerated + fallback) with no python, no torch or cuda.

The [model](https://aihub.qualcomm.com/compute/models/esrgan) used, is from Qualcomm AI Hub as `.onnx` format.

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

The project contains only three files (991 loc):

- [lib](src/lib.rs) - main library file (load_model and run)
- [kernels](src/kernels.rs) - kernel implementations (conv, leaky relu, concat, add, mul and resize)
- [tensor](src/tensor.rs) - basic tensor utilities

> NOTE: Other models or similar variants from different sources from the internet may fail, as the code is hardcoded.

| Input                             | Output (x4)                           | Output (x8)                           |
|-----------------------------------|---------------------------------------|---------------------------------------|
| <img src="input.png" width="512"> | <img src="output_x4.png" width="512"> | <img src="output_x8.png" width="512"> |

```terminaloutput
esrgan --input input.png --output output_x4.png
upscaling 128x128 image, to 512x512 output
progressing: 1/1 tiles processed
finished processing all tiles, took 3.06s

esrgan --input output_x4.png --output output_x8.png
upscaling 512x512 image, to 2048x2048 output
progressing: 16/16 tiles processed
finished processing all tiles, took 46.97s
```

---

Checkout my other [project](https://github.com/ronakgh97/onnx-rs) - minimal onnx-runtime in pure rust