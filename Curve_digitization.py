
''' Convert hand drawn heat flow curves into data that can be processed by the CVAE mode ''''

import cv2
import numpy as np
from skimage.morphology import medial_axis
from scipy import ndimage
from skimage.morphology import skeletonize, remove_small_objects

# 读取图像
# image = cv2.imread("dsc.jpg", 0)
# Parameter 0 indicates reading in grayscale image. If it is a color image, the parameter can be omitted or set to 1
image = cv2.imread("draft-dsc.png",0)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image = cv2.GaussianBlur(image, (3, 3), 0)  # Gaussian blur, whose kernel size and standard deviation can be adjusted as needed

# Perform binary processing
_, binary = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Perform morphological closure operation to fill the voids inside the curve
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Perform central axis transformation
skel, distance = medial_axis(closed, return_distance=True)

# Calculate the centerline based on the distance transformation results
centerline = np.zeros_like(closed, dtype=np.uint8)
centerline[skel] = 255


# Use threshold for binarization to obtain the final centerline
_, result = cv2.threshold(centerline, 0, 255, cv2.THRESH_BINARY)
print(result.shape,result[:100,:100])

## Eliminate burrs
_, labels, stats, _ = cv2.connectedComponentsWithStats(result)

#Filter connected regions based on area
filtered = np.zeros_like(result)
threshold = 255
for i in range(1, len(stats)):
    area = stats[i, cv2.CC_STAT_AREA]
    if area > threshold:  # 根据需要调整阈值
        filtered[labels == i] = 255


# Display centerline results
cv2.imshow('Centerline', result)
# cv2.imshow('Centerline_filter', closed)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Conversion from Pixel to Temperature-Heat Flow
indices = np.where(result > 0)
# print(indices)
rows, cols = indices[0],indices[1]
print(rows.shape,cols.shape)
minc, maxc = min(cols),max(cols)
print(minc, maxc)
dsc_data = np.zeros((len(rows),2))
for i in range(len(rows)):
    dsc_data[i,0] = cols[i]*160/(maxc-minc)
    dsc_data[i,1] = rows[i]*0.4/(max(rows)-min(rows))

print(dsc_data[:10,:])

