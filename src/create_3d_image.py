import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.use('TkAgg')


def reconstruct_3d(rectified_img1, rectified_img2, B, f=32500):
    """
    :param rectified_img1: first rectify image
    :param rectified_img2: second rectified image
    :param f: Focal length of camera [pixels] - default Galaxy A32
    :param B: Baseline distance between two cameras distances [meter]
    """
    # Create a StereoSGBM object
    stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=21)

    # Compute disparity map
    disparity = stereo.compute(cv2.cvtColor(rectified_img1, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(rectified_img2, cv2.COLOR_BGR2GRAY))
    disparity[disparity == 0] = np.mean(disparity)

    plt.imshow(disparity)
    plt.title('Disparity map')
    plt.show()

    depth = f * B / disparity

    plt.imshow(rectified_img1, alpha=0.5)
    plt.imshow(depth, cmap='hot', interpolation='bilinear', alpha=0.5)
    plt.title('Depth heatmap')
    plt.show()

    print('test')
