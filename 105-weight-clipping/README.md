# Weight Clipping for Improved Bit Error Robustness

Training various architectures in the given precision for quantization:

    cd examples/
    python3 train.py --architecture=wrn2810|resnet50|simplenet --directory=output_directory --precision=<precision> --w_max=0.1 --normalization=regn

where `w_max` denotes the maximum absolute weight value used for clipping and `normalization` should be `regn` or `rebn`, corresponding to reparaetermized versions of GN and BN that work with weight clipping.

To evaluate a trained model, simply use

    python3 train.py --directory=output_directory