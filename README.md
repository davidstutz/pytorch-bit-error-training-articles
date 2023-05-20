## PyTorch Examples

This repository contains several PyTorch examples demonstrating fixed-point quantization of deep neural networks, quantization-aware training, bit errors in quantized weights and random bit error training:

* [Simple Fixed-Point Quantization for DNNs in PyTorch](101-network-quantization/README.md)
* [4.5% Test Error on CIFAR10 with 4-Bit Fixed-Point Quantization](102-quantization-aware-training/README.md)
* [Implementing Fast Bitwise Operations for PyTorch](103-bit-operations/README.md)
* [Testing Robustness Against Bit Errors in Quantized DNN Weights](104-bit-errors/README.md)
* [Weight Clipping for Improved Bit Error Robustness](105-weight-clipping/README.md)
* [Random Bit Error Training in PyTorch](106-bit-error-training/README.md)

The examples correspond to an article series on my blog:
[davidstutz.de](https://davidstutz.de/).

The code is partly based on utilities from the examples in
[davidstutz/pytorch-adversarial-examples-training-articles](https://github.com/davidstutz/pytorch-adversarial-examples-training-articles).

Large parts of this repository are taken from my latest
MLSys and TPAMI papers:
    
    [1] D. Stutz, N. Chandramoorthy, M. Hein, B. Schiele.
        Bit Error Robustness for Energy-Efficient DNN Accelerators.
        MLSys, 2021.
    [2] D. Stutz, N. Chandramoorthy, M. Hein, B. Schiele.
        Random and Adversarial Bit Error Robustness: Energy-Efficient and Secure DNN Accelerators.
        TPAMI, 2022

## Installation

Installation is easy with [Conda](https://docs.conda.io/en/latest/):

    conda env create -f environment.yml

You can use `python3 setup.py` to check some of the requirements.
See `environment.yml` for details and versions.

## License

 Copyright (c) 2022 David Stutz, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the corresponding papers (see above) in documents and papers that report on research using the Software.