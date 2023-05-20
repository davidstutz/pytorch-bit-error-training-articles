# Simple Fixed-Point Quantization for DNNs in PyTorch

Quantizing a pre-trained model, for example from [davidstutz/pytorch-adversarial-examples-training-articles](https://github.com/davidstutz/pytorch-adversarial-examples-training-articles):

    cd examples/
    python3 quantize.py ---precision=8 --model_file=path/to/model.pth.tar