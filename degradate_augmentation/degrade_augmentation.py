from keras.preprocessing import image
import numpy as np
import pandas as pd
import cv2


def degrade_image(img, factor):               
    h, w, _ = img.shape
    new_height = int(h / factor)
    new_width = int(w / factor)
    img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR)
    img = cv2.resize(img, (w, h), interpolation = cv2.INTER_LINEAR)
    return img


if __name__ == "__main__":
    img = image.load_img("../input_image.jpg")
    image_matrix = image.img_to_array(img)
    output_img_matrix = degrade_image(image_matrix, 4)
    output_image = image.array_to_img(output_img_matrix)
    output_image.save("degradate_augmentation_output.jpg")
