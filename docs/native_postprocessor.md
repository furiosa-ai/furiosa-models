Some models include the native post-processing implementations,
optimized for FuriosaAI Warboy and Intel/AMD CPUs.

Basically, furiosa-models includes pre/post-processing implementation in Python for each model.
They are reference implementations and can work with CPU and other accelerators like GPU.

The native post processor is implemented in Rust and C++, and works with only FuriosaAI NPU.
The implementation is designed to leverage FuriosaAI NPU's characteristics even for post-processing
and maximize the latency and throughput by using
the characteristics of modern CPU architecture, such as CPU cache, SIMD instructions and CPU pipelining.

*Table 1. Models that support native-postprocessors and their benchmark*

| Model                                   | Latency (Python) | Latency (Native) |
|-----------------------------------------|------------------|------------------|
| [ResNet50](models/resnet50_v1.5.md)     |                  |                  |
| [SSDMobileNet](models/ssd_mobilenet.md) |                  |                  |
| [SSDResNet34](models/ssd_resnet34.md)   |                  |                  |

### Usage

To use native post processor, please pass `use_native_post=True` when a model is initialized.
After then, you need to initialize `NativePostProcessor`.
To evaluate the postprocessing results, please call `NativePostProcessor.eval()`.
The following is an example to use native post processor for [SSDMobileNet](models/ssd_mobilenet.md).

```python
from furiosa.models.vision import SSDMobileNet
from furiosa.models.vision.ssd_mobilenet import preprocess, NativePostProcessor
from furiosa.runtime import session

ssd_mobilenet = SSDMobileNet(use_native_post=True)

postprocessor = NativePostProcessor(ssd_mobilenet)
with session.create(ssd_mobilenet.enf) as sess:
    image, context = preprocess(["tests/assets/cat.jpg"])
    output = sess.run(image).numpy()
    postprocessor.eval(output, context=context)
```