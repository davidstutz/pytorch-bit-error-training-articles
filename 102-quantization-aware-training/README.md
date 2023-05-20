# 4.5% Test Error on CIFAR10 with 4-Bit Fixed-Point Quantization

Training various architectures in the given precision for quantization:

    cd examples/
    python3 train.py --architecture=wrn2810|resnet50|simplenet --directory=output_directory

To evaluate a trained model, simply use

    python3 train.py --directory=output_directory