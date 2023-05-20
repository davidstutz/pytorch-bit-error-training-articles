# Testing Robustness Against Bit Errors in Quantized DNN Weights

To evaluate a model under bit errors in quantized weights:

    cd examples/
    python3 attack.py --model_file=path/to/model.pth.tar --precision=<precision> --p=<bit_error_rate>