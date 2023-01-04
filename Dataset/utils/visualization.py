import cv2
import numpy as np
from matplotlib import pyplot as plt

def draw_matching(image_1, image_2, matching, mask12):
    # draw lines between matching points
    if image_1.dtype != np.uint8:
        image_1 = (image_1 * 255.0).astype(np.uint8)
    if image_2.dtype != np.uint8:
        image_2 = (image_2 * 255.0).astype(np.uint8)

    # matching = matching * mask12[:, :, None]
    matching = matching.astype(np.int32)

    full_image = np.concatenate([image_1, image_2], axis=1)
    full_image = np.ascontiguousarray(full_image, dtype=np.uint8)

    for i in range(matching.shape[0]):
        for j in range(matching.shape[1]):
            if i % 50 == 0 and j % 50 == 0:
                if mask12[i,j] == 0:
                    continue
                src = (j, i)
                dst = (matching[i, j, 0] + image_1.shape[1], matching[i, j, 1])
                rand_color =np.random.randint(0, 255, 3).tolist()
                full_image = cv2.line(full_image, src, dst, rand_color, 2)

    return full_image


def draw_flow(image_1, flow):
    image_flow = image_1.copy()
    if image_1.dtype != np.uint8:
        image_1 = (image_1 * 255.0).astype(np.uint8)

    flow = flow.permute(1, 2, 0).cpu().numpy()
    assert image_1.shape[:2] == flow.shape[:2]
    flow_dire = flow[:, :, :2] / np.linalg.norm(flow[:, :, :2], axis=2, keepdims=True)
    # drwa arrows on img1 according to flow12
    for i in range(20, flow_dire.shape[0], 30):
        for j in range(20, flow_dire.shape[1], 30):
            cv2.arrowedLine(image_flow, (j, i), (int(j+flow_dire[i, j, 0]*15), int(i+flow_dire[i, j, 1]*15)), (255, 0, 0), 2, tipLength=0.4)

    return image_flow
