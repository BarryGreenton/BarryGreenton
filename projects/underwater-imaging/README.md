# Diagnosing underwater image degradation

Should a restoration algorithm correct the colours, lift the shadows or sharpen the edges first?

The 2024 APMCM team used histogram displacement, average grayscale intensity and Laplacian variance to classify 400 images. Its priority rule assigned 222 to colour cast, 107 to low light and 71 to blur. The study compared enhancement methods within each group.

The public example implements the analysis kernels in English. It measures channel imbalance and brightness, estimates edge variation, and applies the submission’s 3 × 3 sharpening kernel. Synthetic input makes the example runnable without the competition photographs.

## Run the example

From this project directory, using Python 3.12 in a virtual environment:

```sh
python -m pip install -r requirements.txt
python example.py
```

[restoration.py](restoration.py) contains the original colour correction, low-light and deblurring functions, extracted from the batch script. Install `opencv-python` alongside the example dependencies to call these functions with OpenCV BGR images. The blur branch sharpens the image, estimates a Gaussian point-spread function and applies Wiener deconvolution.

[Return to the portfolio](../../README.md)
