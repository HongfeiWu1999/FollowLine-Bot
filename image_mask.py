import numpy as np

def getImageMask(img, color):
    return np.all(img == color, axis=-1)

def getImage(img, color):
    mask = getImageMask(img, color)
    image = np.zeros_like(img)
    # asignar el valor blanco (255) solo a los píxeles que cumplen la condición
    image[mask] = 255
    return image

def normalizeImage(image):
    sum = np.sum(image, axis=2)
    return image / sum[:, :, np.newaxis]