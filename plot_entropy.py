import cv2
import numpy as np
import math
import statistics
from tqdm import tqdm
import matplotlib.pyplot as plt



# parameters
image_size = 1000


print('load the image...')
img = cv2.imread('data/DSC_0003.JPG') #path to the image
img = np.float64(img)
# Determine the scaling factor, keeping the aspect ratio
height, width = img.shape[:2]
scaling_factor = image_size / max(height, width)
# Resize the image
resized_img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
# cv2.imwrite('resized_input.png', resized_img)
print('Done!')

# Split channels
blue, green, red = cv2.split(resized_img)
red[red==0] = 1e-6
green[green==0] = 1e-6
blue[blue==0] = 1e-6

# Geometric Mean Invariant Image
geometric_mean = (red * green * blue) ** (1/3)
log_chromaticity_red = np.log(red / geometric_mean)
log_chromaticity_green = np.log(green / geometric_mean)
log_chromaticity_blue = np.log(blue / geometric_mean)

U = np.array([
    [1/np.sqrt(2), -1/np.sqrt(2), 0],
    [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)]
])
rho = np.concatenate(
    (
        np.atleast_3d(log_chromaticity_red),
        np.atleast_3d(log_chromaticity_green),
        np.atleast_3d(log_chromaticity_blue),
    ),
    axis=2
)
X = np.dot(rho, U.T)

# plot 2D log-chromaticity representation
plot_data = X.reshape(-1, 2)

x = plot_data[:, 0]
y = plot_data[:, 1]
plt.scatter(x, y, s=1)
plt.xlabel('X axis label')  # Optionally set the label for the x-axis
plt.ylabel('Y axis label')  # Optionally set the label for the y-axis
plt.show()
