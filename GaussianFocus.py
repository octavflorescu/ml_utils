import PIL
import numpy as np

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_kernel(size, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()
 
    if verbose:
        plt.imshow(kernel_2D, interpolation='none',cmap='gray')
        plt.title("Image")
        plt.show()
 
    return kernel_2D

def bgr_to_rgb(pil_img):
    b, g, r = pil_img.split()
    im = PIL.Image.merge("RGB", (r, g, b))
    
    im_side_size = max(im.size)
    im = im * np.expand_dims(gaussian_kernel(im_side_size, sigma=im_side_size/3), axis=2)[:im.size[1],:im.size[0]]

    return im
