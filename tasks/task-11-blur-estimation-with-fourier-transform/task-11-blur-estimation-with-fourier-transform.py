"""
task-11-blur-estimation-with-fourier-transform.py

#IMPORTANT #
Implement the function `frequency_blur_score` below.

Rules:
- Keep the function name and signature EXACTLY the same.
- Do NOT use any external network calls.
- You may ONLY use standard Python, NumPy, and OpenCV (cv2).
- Return a single float (higher = sharper OR lower = blurrier, but be consistent).

Tip (from the FFT blur-detection tutorial):
- Convert to grayscale
- 2D FFT -> shift DC to center
- Zero-out a centered square (low frequencies)
- Magnitude spectrum (e.g., log1p(abs(...)))
- Use the mean magnitude of the remaining spectrum as the score
"""

from typing import Union
import numpy as np
import cv2


def frequency_blur_score(
    image: Union[np.ndarray, "cv2.Mat"],
    center_size: int = 60
) -> float:
    """
    Compute a blur/sharpness score in the frequency domain.

    Parameters
    ----------
    image : np.ndarray
        Input image, grayscale or BGR. Any dtype accepted; will be converted to float32.
    center_size : int, default=60
        Side length of the central square (low-frequency) region to suppress.

    Returns
    -------
    float
        A scalar score. SHARPER images get a HIGHER score.
    """

    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()


    gray = gray.astype(np.float32)


    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)


    h, w = gray.shape
    cy, cx = h // 2, w // 2
    half = center_size // 2
    fshift[cy - half:cy + half, cx - half:cx + half] = 0


    magnitude = np.abs(fshift)


    magnitude = np.log1p(magnitude)


    score = float(np.mean(magnitude))

    return score
