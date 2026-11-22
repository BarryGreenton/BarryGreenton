import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

def enhance_color_bias_1(image):
    image = image.astype(np.float32) / 255.0
    mean_r, mean_g, mean_b = (np.mean(image[:, :, 2]), np.mean(image[:, :, 1]), np.mean(image[:, :, 0]))
    mean_rgb = (mean_r + mean_g + mean_b) / 3
    image[:, :, 2] = image[:, :, 2] * (mean_rgb / mean_r)
    image[:, :, 1] = image[:, :, 1] * (mean_rgb / mean_g)
    image[:, :, 0] = image[:, :, 0] * (mean_rgb / mean_b)
    image = np.clip(image, 0, 1)
    enhanced_image = (image * 255).astype(np.uint8)
    lab = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced_image


def enhance_color_bias_2(image):
    image = image.astype(np.float32) / 255.0
    mean_r, mean_g, mean_b = (np.mean(image[:, :, 2]), np.mean(image[:, :, 1]), np.mean(image[:, :, 0]))
    mean_values = [mean_r, mean_g, mean_b]
    max_channel_index = np.argmax(mean_values)
    target_channel = image[:, :, max_channel_index]
    for i in range(3):
        if i != max_channel_index:
            image[:, :, i] = histogram_matching(image[:, :, i], target_channel)
    hsv_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    intensity = hsv_image[:, :, 2] / 255.0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    intensity = clahe.apply((intensity * 255).astype(np.uint8))
    intensity = np.clip(intensity, 0, 255)
    hsv_image[:, :, 2] = intensity
    enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return enhanced_image.astype(np.uint8)


def histogram_matching(source, reference):
    source = (source * 255).astype(np.uint8)
    reference = (reference * 255).astype(np.uint8)
    source_hist, bins = np.histogram(source.flatten(), 256, [0, 256])
    reference_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
    cdf_source = np.cumsum(source_hist).astype(np.float32) / source_hist.sum()
    cdf_reference = np.cumsum(reference_hist).astype(np.float32) / reference_hist.sum()
    lookup_table = np.zeros(256, dtype=np.uint8)
    for gray_value in range(256):
        lookup_table[gray_value] = np.searchsorted(cdf_reference, cdf_source[gray_value])
    matched = lookup_table[source]
    return matched.astype(np.float32) / 255.0


def enhance_low_light_1(image):
    image = image.astype(np.float32) / 255.0

    def compute_dark_channel(img, size=15):
        min_channel = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel
    dark_channel = compute_dark_channel(image)

    def estimate_atmospheric_light(img, dark_channel, top_percent=0.1):
        num_pixels = int(top_percent * dark_channel.size)
        dark_vec = dark_channel.ravel()
        indices = np.argsort(dark_vec)[-num_pixels:]
        atmospheric_light = np.mean(img.reshape(-1, 3)[indices], axis=0)
        return atmospheric_light
    atmospheric_light = estimate_atmospheric_light(image, dark_channel)
    omega = 0.95
    transmission = 1 - omega * dark_channel

    def restore_image(img, t, A, t_min=0.1):
        t = np.maximum(t, t_min)
        t = np.expand_dims(t, axis=2)
        J = (img - A) / t + A
        return np.clip(J, 0, 1)
    enhanced_image = restore_image(image, transmission, atmospheric_light)
    gamma = 1.2
    enhanced_image = np.power(enhanced_image, 1 / gamma)
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    return enhanced_image


def enhance_low_light_2(image):
    image = image.astype(np.float32) / 255.0
    gamma = 2.2
    gamma_corrected = np.power(image, 1 / gamma)
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced_image


def enhance_low_light_3(image, scales=[8, 16, 32]):
    yuv_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_channel = yuv_img[:, :, 0]
    enhanced_y = np.zeros_like(y_channel, dtype=np.float32)
    for scale in scales:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(scale, scale))
        local_enhanced = clahe.apply(y_channel)
        enhanced_y += local_enhanced.astype(np.float32) / len(scales)
    enhanced_y = np.clip(enhanced_y, 0, 255).astype(np.uint8)
    yuv_img[:, :, 0] = enhanced_y
    enhanced_image = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
    return enhanced_image


def enhance_blur_scene_1(image):

    def laplacian_sharpening(img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    sharpened_image = laplacian_sharpening(image)

    def estimate_blur_kernel(size=15, sigma=3):
        ax = np.linspace(-(size // 2), size // 2, size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        return kernel / np.sum(kernel)
    blur_kernel = estimate_blur_kernel(size=15, sigma=3)

    def wiener_deconvolution(img, kernel, K=10):
        if len(img.shape) == 3:
            deblurred_channels = []
            for c in range(img.shape[2]):
                img_fft = np.fft.fft2(img[:, :, c])
                kernel_fft = np.fft.fft2(kernel, s=img.shape[:2])
                kernel_fft = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
                deblurred_fft = img_fft * kernel_fft
                deblurred = np.fft.ifft2(deblurred_fft)
                deblurred_channels.append(np.abs(deblurred))
            return np.stack(deblurred_channels, axis=2)
        else:
            img_fft = np.fft.fft2(img)
            kernel_fft = np.fft.fft2(kernel, s=img.shape)
            kernel_fft = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
            deblurred_fft = img_fft * kernel_fft
            deblurred = np.fft.ifft2(deblurred_fft)
            return np.abs(deblurred)
    deblurred_image = wiener_deconvolution(sharpened_image, blur_kernel)

    def adjust_contrast(img, alpha=1.5, beta=20):
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    enhanced_image = adjust_contrast(deblurred_image)
    return enhanced_image
