import cv2
import numpy as np
from matplotlib import pyplot as plt

def draw_matching(image_1, image_2, matching, mask12):
    # draw lines between matching points
    if image_1.dtype != np.uint8:
        image_1 = (image_1 * 255.0).astype(np.uint8)
    if image_2.dtype != np.uint8:
        image_2 = (image_2 * 255.0).astype(np.uint8)

    matching = matching * mask12[:, :, None]
    matching = matching.astype(np.int32)

    full_image = np.concatenate([image_1, image_2], axis=1)
    full_image = np.ascontiguousarray(full_image, dtype=np.uint8)

    for i in range(matching.shape[0]):
        for j in range(matching.shape[1]):
            if i % 50 == 0 and j % 50 == 0:
                src = (j, i)
                dst = (matching[i, j, 0] + image_1.shape[1], matching[i, j, 1])
                rand_color =np.random.randint(0, 255, 3).tolist()
                full_image = cv2.line(full_image, src, dst, rand_color, 2)

    return full_image
