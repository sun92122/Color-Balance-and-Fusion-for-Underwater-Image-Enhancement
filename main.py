import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from tools import to_float, to_uint8, save_image, show_buffered_images, push_image_to_buffer


# === Global Variables ===
image_buffer = []

# === Config ===
# 是否顯示中間結果
DEBUG = True
# 色彩補償係數
ALPHA = 2.5
# Gamma 值
GAMMA = 4
# Unsharp Masking 的高斯模糊強度
SIGMA = 5
# 金字塔層數
LEVELS = 7
# 是否使用藍通道補償
BLUE_COMPENSATION = False
# 白平衡增益
WHITE_BALANCE_GAIN = 1.2


# === Functions ===
def gamma_correction(img, gamma):
    """
    Gamma correction for image enhancement.

    parameters
    ----------
    img:
        RGB image
    gamma:
        Gamma value

    returns
    -------
    img:
        Gamma corrected image
    """
    return np.power(img, gamma)


def normalized_unsharp_mask(img, sigma):
    """
    Normalized unsharp masking for image enhancement.

    parameters
    ----------
    img:
        RGB image
    sigma:
        Gaussian filter standard deviation

    returns
    -------
    img:
        Enhanced image
    """
    blurred = gaussian_filter(img, sigma=sigma)
    highpass = img - blurred
    # Histogram stretching for highpass
    p2, p98 = np.percentile(highpass, (2, 98))
    highpass = np.clip((highpass - p2) / (p98 - p2 + 1e-8), 0, 1)
    return 0.5 * (img + highpass)


def red_channel_compensation(img, alpha=1.0):
    """
    Red channel compensation using gray world assumption.

    parameters
    ----------
    img:
        RGB image
    alpha:
        Compensation factor

    returns
    -------
    img:
        Compensated image
    """
    Ir, Ig, _ = cv2.split(img)
    Ir_mean, Ig_mean = np.mean(Ir), np.mean(Ig)
    delta = (Ig_mean - Ir_mean) * (1 - Ir) * Ig
    Ir_comp = Ir + alpha * delta
    Ir_comp = np.clip(Ir_comp, 0, 1)
    result = img.copy()
    result[..., 0] = Ir_comp
    return result


# experimental, not used
def red_channel_compensation_v2(img, alpha=1.0):
    """
    Red channel compensation version 2, with both green and blue channels

    parameters
    ----------
    img:
        RGB image
    alpha:
        Compensation factor

    returns
    -------
    img:
        Compensated image
    """
    Ir, Ig, Ib = cv2.split(img)
    Igb = (Ig + Ib) / 2
    Ir_mean, Igb_mean = np.mean(Ir), np.mean(Igb)
    delta = (Igb_mean - Ir_mean) * (1 - Ir) * Igb
    Ir_comp = Ir + alpha * delta
    Ir_comp = np.clip(Ir_comp, 0, 1)
    result = img.copy()
    result[..., 0] = Ir_comp
    return result


def blue_channel_compensation(img, alpha=1):
    _, Ig, Ib = cv2.split(img)
    Ib_mean, Ig_mean = np.mean(Ib), np.mean(Ig)
    delta = (Ig_mean - Ib_mean) * (1 - Ib) * Ig
    Ib_comp = Ib + alpha * delta
    Ib_comp = np.clip(Ib_comp, 0, 1)
    result = img.copy()
    result[..., 2] = Ib_comp
    return result


# experimental, not used
def lab_balance(img_rgb):
    img_bgr = cv2.cvtColor(to_uint8(img_rgb), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 移除色偏（a, b 通道偏離 128 代表偏紅綠或偏藍黃）
    a = cv2.addWeighted(a, 1, 128 * np.ones_like(a), 0, -np.mean(a) + 128)
    b = cv2.addWeighted(b, 1, 128 * np.ones_like(b), 0, -np.mean(b) + 128)

    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return to_float(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def white_balance(img, gain=1.0):
    """
    White balance using gray world assumption.

    parameters
    ----------
    img:
        RGB image
    gain:
        Gain factor for white balance

    returns
    -------
    wb_img:
        White balanced image
    """
    img_c = red_channel_compensation(img, alpha=ALPHA)
    if DEBUG:
        push_image_to_buffer(image_buffer)
        push_image_to_buffer(image_buffer, img_c, "Red Channel Compensation")

    if BLUE_COMPENSATION:
        img_c = blue_channel_compensation(img_c, alpha=ALPHA)
        if DEBUG:
            push_image_to_buffer(image_buffer, img_c,
                                 "Blue Channel Compensation")
    else:
        push_image_to_buffer(image_buffer)

    mean_vals = np.mean(img_c, axis=(0, 1))  # R, G, B 的均值
    # mean_val = np.mean(mean_vals)  # 三個通道的均值
    mean_val = np.sum(np.array([0.299, 0.587, 0.114]) * mean_vals)  # 加權平均
    scale = mean_val / mean_vals  # 計算縮放比例
    wb_img = img_c * scale  # 白平衡
    wb_img = np.clip(wb_img * gain, 0, 1)  # 限制在 [0, 1] 範圍內
    return wb_img


def laplacian_weight_map(gray_img):
    """
    parameters
    ----------
    gray_img:
        grayscale image

    returns
    -------
    lap:
        Laplacian weight map
    """
    lap = cv2.Laplacian(gray_img, cv2.CV_64F)
    lap = np.clip(np.abs(lap), 0, 1)
    return lap


def saliency_weight_map(gray_img, w_hb):
    """
    parameters
    ----------
    gray_img:
        grayscale image
    w_hb:

    returns
    -------
    saliency_map:
        Saliency map
    """
    # 1. Gaussian filter
    gauss_img = cv2.GaussianBlur(gray_img, (11, 11), w_hb)
    # 2. Mean value
    img_mean = np.mean(gray_img)
    # 3. Saliency map
    saliency_map = np.clip(np.abs(gauss_img - img_mean), 0, 1)
    return saliency_map


def saturation_weight_map(img):
    """
    parameters
    ----------
    img:
        RGB image

    returns
    -------
    WSat:
        Saturation weight map
    """
    R, G, B = cv2.split(img)
    L = (R + G + B) / 3
    WSat = np.sqrt((R - L) ** 2 + (G - L) ** 2 + (B - L) ** 2) / np.sqrt(3)
    WSat = np.clip(WSat, 0, 1)
    return WSat


def compute_weights(img):
    """
    Compute weight maps for the image.

    parameters
    ----------
    img:
        RGB image

    returns
    -------
    W:
        Combined weight map
    """
    gray = cv2.cvtColor(to_uint8(img), cv2.COLOR_BGR2GRAY) / 255.0

    # Laplacian contrast
    WL = laplacian_weight_map(gray)
    if DEBUG:
        push_image_to_buffer(
            image_buffer, WL, "Laplacian Contrast", cmap='gray')

    # Saliency (Achanta method approximation)
    WS = saliency_weight_map(gray, np.pi/2.75)
    if DEBUG:
        push_image_to_buffer(image_buffer, WS, "Saliency", cmap='gray')

    # Saturation
    WSat = saturation_weight_map(img)
    if DEBUG:
        push_image_to_buffer(image_buffer, WSat, "Saturation", cmap='gray')

    W = WL + WS + WSat
    return W + 1e-8  # avoid division by zero


def build_laplacian_pyramid(img, levels):
    gp = [img]
    for _ in range(levels - 1):
        img = cv2.pyrDown(img)
        gp.append(img)
    lp = []
    for i in range(levels - 1):
        size = (gp[i].shape[1], gp[i].shape[0])
        GE = cv2.pyrUp(gp[i + 1], dstsize=size)
        lap = gp[i] - GE
        lp.append(lap)
    lp.append(gp[-1])  # 最後一層為最底層的高斯圖
    return lp


def build_gaussian_pyramid(weight, levels):
    gp = [weight]
    for _ in range(levels - 1):
        weight = cv2.pyrDown(weight)
        gp.append(weight)
    return gp


def multiscale_fusion(img1, img2, w1, w2, levels=5):
    """
    Multi-scale fusion of two images using Laplacian pyramids and Gaussian pyramids.

    parameters
    ----------
    img1:
        First input image
    img2:
        Second input image
    w1:
        Weight map for the first image
    w2:
        Weight map for the second image
    levels:
        Number of levels in the pyramid

    returns
    -------
    fused_image:
        Fused image
    """
    # 建立拉普拉斯金字塔
    lp1 = build_laplacian_pyramid(img1, levels)
    lp2 = build_laplacian_pyramid(img2, levels)

    # 建立高斯金字塔（權重圖）
    gpW1 = build_gaussian_pyramid(w1, levels)
    gpW2 = build_gaussian_pyramid(w2, levels)

    fused_pyramid = []

    delta = 1e-8  # 避免除以零
    for i in range(levels):
        # 對權重圖進行正規化（Eq. 10）
        W1n = gpW1[i]
        W2n = gpW2[i]
        w_sum = W1n + W2n + delta
        w1n = W1n / w_sum
        w2n = W2n / w_sum

        # 融合該層
        w1n = np.expand_dims(w1n, axis=2)
        w2n = np.expand_dims(w2n, axis=2)
        blended = w1n * lp1[i] + w2n * lp2[i]
        fused_pyramid.append(blended)

    # 重建圖像
    result = fused_pyramid[-1]
    for i in range(levels - 2, -1, -1):
        size = (fused_pyramid[i].shape[1], fused_pyramid[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size)
        result = result + fused_pyramid[i]

    return np.clip(result, 0, 1)


def show_image(img, title="Image", cmap=None):
    """
    Show image using matplotlib

    parameters
    ----------
    img:
        RGB image
    title:
        image title
    cmap:
        colormap (default None)
    """
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()


# === main function ===
def enhance_underwater_image(img):

    if DEBUG:
        push_image_to_buffer(image_buffer, img, "Original Image")

    # Step 1: 色彩補償（紅通道）
    # Step 2: 白平衡（灰世界）
    wb_img = white_balance(img, gain=WHITE_BALANCE_GAIN)
    if DEBUG:
        push_image_to_buffer(image_buffer, wb_img, "White Balanced Image")

    # Step 3: 兩個輸入
    # Step 4: 權重圖
    input1 = gamma_correction(wb_img, GAMMA)
    if DEBUG:
        push_image_to_buffer(image_buffer, input1, "Gamma Corrected Image")
    W1 = compute_weights(input1)
    if DEBUG:
        push_image_to_buffer(image_buffer, W1, "Weight Image 1", cmap='gray')

    input2 = normalized_unsharp_mask(wb_img, sigma=SIGMA)
    if DEBUG:
        push_image_to_buffer(image_buffer, input2, "Unsharp Masked Image")
    W2 = compute_weights(input2)
    if DEBUG:
        push_image_to_buffer(image_buffer, W2, "Weight Image 2", cmap='gray')

    # Step 5: 多尺度融合
    result = multiscale_fusion(input1, input2, W1, W2, levels=LEVELS)

    return to_uint8(result)


# === 範例使用 ===
if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = ".\\tests\\underwater_sample.jpg"

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        sys.exit(1)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = to_float(img)

    result = enhance_underwater_image(img)
    if DEBUG:
        push_image_to_buffer(image_buffer, result, "Final Result")
        show_buffered_images(
            image_buffer,
            f"Alpha={ALPHA}, WB_Gain={WHITE_BALANCE_GAIN}, Blue_Comp={BLUE_COMPENSATION}, Gamma={GAMMA}, Sigma={SIGMA}, Levels={LEVELS}",
            save_path=f"{img_path.removesuffix('.'+img_path.split('.')[-1])}_result.jpg")
    else:
        show_image(result, "Final Result")
