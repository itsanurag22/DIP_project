# This is a sample Python script.
import numpy as np
import cv2
import math


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def convolve2D(image,kernel):
    """
    For convolution of 2D-image and 2D-kernel
    """
    kernel = np.flipud(np.fliplr(kernel))
    #Dimensions of the image as well as the kernel
    x_img = image.shape[0]
    y_img = image.shape[1]
    x_kernel = kernel.shape[0]
    y_kernel = kernel.shape[1]
    padding = x_kernel//2
    #Dimensions of output image
    xOutput = int((x_img - x_kernel +(2*padding) + 1))
    yOutput = int((y_img - y_kernel +(2 * padding)+ 1))
    output = np.zeros((xOutput, yOutput))

    if padding!=0:
        imagePadded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    for x in range(image.shape[0]):

        for y in range(image.shape[1]):

            try:
                #Element wise multiplication of matrices
                sum=0
                for i in range(x_kernel):
                    for j in range(y_kernel):
                        sum = sum + kernel[i][j]*imagePadded[x+i][y+j]
                output[x][y] = sum
            except:
                break

    return output



def convolve3D(image,kernel):
    """
    For convolution of 3D-image and 2D-kernel
    """
    kernel = np.flipud(np.fliplr(kernel))
    #Dimensions of the image as well as the kernel
    x_img = image.shape[0]
    y_img = image.shape[1]
    z_img = image.shape[2]
    x_kernel = kernel.shape[0]
    y_kernel = kernel.shape[1]
    padding = x_kernel//2
    # Dimensions of output image
    xOutput = int((x_img - x_kernel +(2*padding) + 1))
    yOutput = int((y_img - y_kernel +(2 * padding)+ 1))
    output = np.zeros((xOutput, yOutput, z_img))

    if padding!=0:
        imagePadded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2, z_img))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image


    for z in range(image.shape[2]):

        for y in range(image.shape[1]):

            for x in range(image.shape[0]):
                try:
                    # Element wise multiplication of matrices
                    sum = 0
                    for i in range(x_kernel):
                        for j in range(y_kernel):
                            sum = sum + kernel[i][j] * imagePadded[x + i][y + j][z]
                    output[x][y][z] = sum
                except:
                    break

    return output

def gaussian_kernel(size=3, sigma=1.):
    """
    Creates gaussian kernel with size = `size` and standard deviation = `sigma`
    """
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    sum = np.sum(kernel)

    return kernel / sum

def avg_kernel(size=3):
    """
    Creates a mean filter with size = `size`
    """
    ones = np.ones((size, size))
    sum = np.sum(ones)

    return ones / sum

def unsharp_masking(image, kernel_size):
    """
    For unsharp masking which sharpens the image
    """
    avg =avg_kernel(kernel_size)
    blurred_image = convolve3D(image, avg)
    mask = img - blurred_image
    sharpened_img = img + mask

    return sharpened_img


def median_filtering(image, kernel_size):
    """
    For median filtering of image with given kernel_size,
    efficiently removes salt pepper noise
    """

    # Dimensions of the image as well as the kernel
    x_img = image.shape[0]
    y_img = image.shape[1]
    z_img = image.shape[2]
    x_kernel = kernel_size
    y_kernel = kernel_size
    padding = x_kernel // 2
    # Dimensions of output image
    xOutput = int((x_img - x_kernel + (2 * padding) + 1))
    yOutput = int((y_img - y_kernel + (2 * padding) + 1))
    output = np.zeros((xOutput, yOutput, z_img))

    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2, z_img))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    for z in range(image.shape[2]):

        for y in range(image.shape[1]):

            for x in range(image.shape[0]):

                try:
                    # Finding the median of sub matrix of size = `kernel_size`
                    med_array = imagePadded[x:x+kernel_size, y:y+kernel_size, z]
                    median = np.median(med_array)
                    output[x][y][z] = median
                except:
                    break

    return output

def conservative_filter(image, kernel_size):
    """
    For conservative filtering of image with given kernel_size,
    removes salt pepper noise but not as good as median filtering
    """
    # Dimensions of the image as well as the kernel
    output = image.copy()
    lim = kernel_size //2
    x_img = image.shape[0]
    y_img = image.shape[1]
    z_img = image.shape[2]
    temp = []
    for z in range(image.shape[2]):

        for x in range(image.shape[0]):

            for y in range(image.shape[1]):

                try:
                    for k in range(x - lim, x + lim + 1):
                        for m in range(y - lim, y + lim + 1):
                            if (k > -1) and (k < x_img):
                                if (m > -1) and (m < y_img):
                                    temp.append(image[k][m][z])

                    temp.remove(image[x][y][z])
                    max_val = max(temp)
                    min_val = min(temp)
                    if image[x][y][z] > max_val:
                        output[x][y][z] = max_val
                    elif image[x][y][z] < min_val:
                        output[x][y][z] = min_val
                    temp=[]
                except:
                    break

    return output




def log_transform(c, f):
    """
    For finding log used in log transform
    """
    g = c * math.log(float(1 + f),10)
    return g

def log_transformation(image, inputMax=255, outputMax=255):
    """
     s = T(r) = c*log(r+1)
     s is the output image
     r is the input image
    """
    c = outputMax / math.log(inputMax + 1, 10)
    output=image.copy()
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # apply log function on the individual rgb values
            bluePixel = round(log_transform(c, image[x][y][0]))
            greenPixel = round(log_transform(c, image[x][y][1]))
            redPixel = round(log_transform(c, image[x][y][2]))

            #assign values to output image
            output[x][y][0] = bluePixel
            output[x][y][1] = greenPixel
            output[x][y][2] = redPixel

    return output


# Read image -------------------------------------------
img = cv2.imread('salt_noise.jpg')

gray_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
print(img)

# Gaussian denoising -----------------------------------
gauss = gaussian_kernel(5,1) #(size, sigma)
# print(gauss_kernel)
# result = convolve3D(img, gauss)

# Mean filtering ----------------------------------------
avg = avg_kernel(5)
# print(avg)
# result = convolve3D(img, avg)

# Edge detection using laplacian--------------------------
lap_kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

lap_kernel_diag =  np.array([[-1, -1, -1],
                             [-1, 8, -1],
                             [-1, -1, -1]])
# result = convolve3D(img, lap_kernel_diag)

# Edge detection using Sobel filters ----------------------
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

# result = convolve3D(img, sobel_x)
# result = convolve3D(img, sobel_y)

# Identity kernel gives the same image as output ----------
identity_kernel = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])

# result = convolve3D(img, identity_kernel)

# Log transformation --------------------------------------
result = log_transformation(img)

#Salt pepper noise removal ---------------------------------
result = conservative_filter(img, 3)
# result = median_filtering(img, 3)

print(result)
cv2.imwrite('result.png', result)










# See PyCharm help at https://www.jetbrains.com/help/pycharm/
