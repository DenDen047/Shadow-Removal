import cv2
import numpy as np
import math
import statistics
from tqdm import tqdm
import matplotlib.pyplot as plt


def normalize_image(img):
    normalized_image = (img - img.min()) / (img.max() - img.min())
    return normalized_image

def save_image_from_arrays(file_name, red, green, blue):
    bgr_img = np.concatenate(
        (
            np.atleast_3d(blue),
            np.atleast_3d(green),
            np.atleast_3d(red),
        ),
        axis=2
    )
    bgr_norm_img = normalize_image(bgr_img)
    cv2.imwrite(file_name, (bgr_norm_img * 255).astype(np.uint8))


# parameters
image_size = 1000


# load the image
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
h, w, c = resized_img.shape
blue, green, red = cv2.split(resized_img)
red += 1e-6
green += 1e-6
blue += 1e-6

# Geometric Mean Invariant Image
geometric_mean = (red * green * blue) ** (1/3)
log_chromaticity_red = np.log(red / geometric_mean)
log_chromaticity_green = np.log(green / geometric_mean)
log_chromaticity_blue = np.log(blue / geometric_mean)
save_image_from_arrays(
    'geometric_mean_invariant.png',
    red / geometric_mean, green / geometric_mean, blue / geometric_mean
)

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

# greyscale invariant images
e_t = np.zeros((2, 181))
for angle in tqdm(range(181)):
    theta = angle * np.pi / 180.0
    e_t[0, angle] = np.cos(theta)
    e_t[1, angle] = np.sin(theta)
I1d = np.dot(X, e_t)

# bin width of probability density histogram
bin_width = np.zeros((181))
for i in tqdm(range(181)):
    bin_width[i] = 3.5 * np.std(I1d[:,:,i]) / (h * w)**(1/3)

# get the angle having the minimum information potential
entropies = []
information_potentials = []
for i in tqdm(range(181)):
    # number of bins
    comp1 = np.mean(I1d[:,:,i]) - 3 * np.std(I1d[:,:,i])
    comp2 = np.mean(I1d[:,:,i]) + 3 * np.std(I1d[:,:,i])
    temp = I1d[:, :, i][(comp1 < I1d[:, :, i]) & (I1d[:, :, i] < comp2)]
    nbins = round((max(temp) - min(temp)) / bin_width[i])
    # histogram
    hist, _ = np.histogram(temp, bins=nbins)
    hist = filter(lambda x: x != 0, hist)
    hist = np.array([float(var) for var in hist])
    pdh = hist / np.sum(hist)   # probability density histogram

    entropies.append(-np.log(np.sum(pdh ** 2)))
    information_potentials.append(np.sum(pdh ** 2))

angle = information_potentials.index(max(information_potentials))
theta = angle * np.pi / 180.0
e_t = np.array([[np.cos(theta), np.sin(theta)]])
e = np.array([[-np.sin(theta), np.cos(theta)]])

# the grayscale invariant image
grey_invariant_img = np.dot(X, e_t.T)
save_image_from_arrays(
    'greyscale_invariant.png',
    grey_invariant_img, grey_invariant_img, grey_invariant_img
)

# 3-Vector Representation
p_th = np.dot(e_t.T, e_t)
X_th = np.dot(X, p_th.T)
mX = np.dot(X, e.T)
mX_th=np.dot(X_th, e.T)

mX=np.atleast_3d(mX)
mX_th=np.atleast_3d(mX_th)

theta=(math.pi*float(angle))/180.0
theta=np.array([
    [np.cos(theta), np.sin(theta)],
    [-1*np.sin(theta),np.cos(theta)]
])
alpha=theta[0,:]
alpha=np.atleast_2d(alpha)
beta=theta[1,:]
beta=np.atleast_2d(beta)

#Finding the top 1% of mX
mX1=mX.reshape(mX.shape[0]*mX.shape[1])
mX1sort=np.argsort(mX1)[::-1]
mX1sort=mX1sort+1
mX1sort1 = np.remainder(mX1sort,mX.shape[1])
mX1sort1=mX1sort1-1
mX1sort2 = (mX1sort / mX.shape[1]).astype(int)
mX_index=[[x,y,0] for x, y in zip(list(mX1sort2), list(mX1sort1))]
mX_top = [
    mX[x[0], x[1], x[2]]
    for x in mX_index[:int(0.01 * mX.shape[0] * mX.shape[1])]
]
mX_th_top=[mX_th[x[0],x[1],x[2]] for x in mX_index[:int(0.01*mX_th.shape[0]*mX_th.shape[1])]]
X_E=(statistics.median(mX_top)-statistics.median(mX_th_top))*beta.T
X_E=X_E.T

for i in tqdm(range(X_th.shape[0])):
   for j in range(X_th.shape[1]):
       X_th[i,j,:]=X_th[i,j,:]+X_E

rho_ti=np.dot(X_th,U)
c_ti=np.exp(rho_ti)
sum_ti=np.sum(c_ti,axis=2)
sum_ti=sum_ti.reshape(c_ti.shape[0],c_ti.shape[1],1)
r_ti=c_ti/sum_ti

invariant_img = (r_ti * 255).astype(np.uint8)


cv2.imwrite(
    'invariant.png',
    cv2.cvtColor(invariant_img, cv2.COLOR_RGB2BGR)
)
