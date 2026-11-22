import numpy as np
from scipy.ndimage import convolve

def image_features(rgb):
    """Compute colour bias, average grayscale intensity and Laplacian variance."""
    rgb = np.asarray(rgb, dtype=float)
    gray = rgb @ np.array([.299, .587, .114])
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    bias = np.abs(rgb.mean(axis=(0, 1)) - gray.mean()).sum()
    return bias, gray.mean(), convolve(gray, laplacian, mode="mirror").var()

def sharpen(rgb):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    channels = [convolve(channel.astype(float), kernel, mode="mirror") for channel in rgb.transpose(2, 0, 1)]
    return np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)

if __name__ == "__main__":
    image = np.full((256, 256, 3), 40, dtype=np.uint8)
    image[:, 128:] = [30, 60, 140]
    enhanced = sharpen(image)
    assert enhanced.shape == image.shape and np.any(enhanced != image)
    print("Bias, brightness, Laplacian variance:", image_features(image))
