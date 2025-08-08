import cv2
import numpy as np

def remove_salt_and_pepper_noise(image: np.ndarray) -> np.ndarray:
    """
    Removes salt and pepper noise from a grayscale image using median filtering.

    Parameters:
        image (np.ndarray): Noisy input image (grayscale).

    Returns:
        np.ndarray: Denoised image.
    """

    denoised_image = cv2.medianBlur(image, ksize=3)
    return denoised_image

if __name__ == "__main__":

    image_path = "../../img/head.png"
    noisy_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if noisy_image is None:
        raise FileNotFoundError(f"A imagem '{image_path}' não foi encontrada.")

    denoised_image = remove_salt_and_pepper_noise(noisy_image)
    cv2.imwrite("denoised_image.png", denoised_image)
