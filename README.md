# Color Balance and Fusion for Underwater Image Enhancement

C. O. Ancuti, C. Ancuti, C. De Vleeschouwer and P. Bekaert, "Color Balance and Fusion for Underwater Image Enhancement," in IEEE Transactions on Image Processing, vol. 27, no. 1, pp. 379-393, Jan. 2018, doi: 10.1109/TIP.2017.2759252.  
Abstract: We introduce an effective technique to enhance the images captured underwater and degraded due to the medium scattering and absorption. Our method is a single image approach that does not require specialized hardware or knowledge about the underwater conditions or scene structure. It builds on the blending of two images that are directly derived from a color-compensated and white-balanced version of the original degraded image. The two images to fusion, as well as their associated weight maps, are defined to promote the transfer of edges and color contrast to the output image. To avoid that the sharp weight map transitions create artifacts in the low frequency components of the reconstructed image, we also adapt a multiscale fusion strategy. Our extensive qualitative and quantitative evaluation reveals that our enhanced images and videos are characterized by better exposedness of the dark regions, improved global contrast, and edges sharpness. Our validation also proves that our algorithm is reasonably independent of the camera settings, and improves the accuracy of several image processing applications, such as image segmentation and keypoint matching.
keywords: {Image color analysis;Scattering;Cameras;Atmospheric modeling;Absorption;Image restoration;Image edge detection;Underwater;image fusion;white-balancing},
URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8058463&isnumber=8071125

---

Final project of NTNU 113-2 Image Processing course  

## 使用

```bath
python main.py {image path}
```

- image path: 水下影像路徑，預設為 `./tests/underwater_sample.jpg`  

### 參數調整

```python
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
```

#### 與原作差異

ALPHA = 1 根本作不出能用的圖  
加入白平衡增益，處理可能源自亮度不足的色偏  
