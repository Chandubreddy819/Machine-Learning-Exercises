import numpy as np
from PIL import Image


def make_kernel(ksize, sigma):
    center_of_kernel = ksize // 2
    u = np.arange(0, ksize, 1)
    v = np.arange(0, ksize, 1)

    u, v = np.meshgrid(u, v)

    u -= center_of_kernel
    v -= center_of_kernel

    gauss = (1 / (2 * np.pi * sigma ** 2)) * np.exp(- (u ** 2 + v ** 2) / (2 * sigma ** 2))

    gauss_kernel = gauss / np.sum(gauss)

    return gauss_kernel


def slow_convolve(arr, k):
    image_height, image_width = arr.shape
    kernel_height, kernel_width = k.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    flipped_k = np.flipud(np.fliplr(k))

    padded_image = np.pad(arr, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    convoluted_image = np.zeros_like(arr)

    for i in range(image_height):
        for j in range(image_width):
            cum_sum = 0
            for u in range(kernel_height):
                for v in range(kernel_width):
                    cum_sum += flipped_k[u, v] * padded_image[i+u, j+v]
            convoluted_image[i, j] = cum_sum

    return convoluted_image


if __name__ == '__main__':
    k = make_kernel(3, 1)  # todo: find better parameters

    # TODO: chose the image you prefer
    im = np.array(Image.open('input1.jpg'))
    # im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))

    im_gray = np.array(Image.fromarray(im).convert('L'))

    result = slow_convolve(im_gray, k)

    output_image = np.clip(result, 0, 255).astype(np.uint8)
    final_image = im_gray + (im_gray - output_image)
    Image.fromarray(final_image).save('output.png')

    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
