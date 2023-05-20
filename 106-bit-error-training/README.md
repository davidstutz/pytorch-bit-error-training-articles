# Random Bit Error Training in PyTorch

Training various architectures in the given precision for quantization:

    cd examples/
    python3 train.py --architecture=wrn2810|resnet50|simplenet --directory=output_directory --precision=<precision> --w_max=0.1 --normalization=regn --p=0.01

where `w_max` denotes the maximum absolute weight value used for clipping, `normalization` should be `regn` or `rebn` and `p` is the bit error rate used during training.

To evaluate a trained model, simply use

    python3 train.py --directory=output_directory