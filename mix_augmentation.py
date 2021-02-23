import random
from keras.preprocessing import image

from bias_augmentaton.bias_augmentaton import image_bias
from brightness_augmentation.brightness_augmentation import brightness
#from deformation_augmentation.deformation_augmentation import deformate_image
from noise_augmentation.noise_augmentation import noise
from scaling_augmentation.scaling_augmentation import scaling_image
from rotation_augmentation.rotation_augmentation import rotate_image

def mix_effects(image_matrix):
    image_matrix = brightness(image_matrix, random.uniform(0.4, 1.2))
    image_matrix = rotate_image(image_matrix, random.randrange(-30, 30))
    image_matrix = noise(image_matrix, random.randrange(0, 40))
    image_matrix = image_bias(image_matrix, random.randrange(-30, 30), random.randrange(-30, 30))
    image_matrix = scaling_image(image_matrix, random.uniform(0.8, 1.2))
    return image_matrix


if __name__ == "__main__":
    img = image.load_img("input_image.jpg")

    image_matrix = image.img_to_array(img)

    output_img_matrix = mix_effects(image_matrix)
    output_image = image.array_to_img(output_img_matrix)

    output_image.save("mix.jpg")