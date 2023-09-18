import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# global variables
pattern_position_tl = {}
pattern_position_br = {}

x_position = 0
y_position = 0


def update_image(img):
    """
    Update image with vertical and horizontal line for easy pattern coordinate finding
    """
    image_copy = img.copy()

    # Draw vertical and horizontal lines based on mouse cursor position
    cv2.line(image_copy, (x_position, 0), (x_position, img.shape[0]), (0, 255, 0), 4)
    cv2.line(image_copy, (0, y_position), (img.shape[1], y_position), (0, 255, 0), 4)

    plt.clf()
    plt.imshow(image_copy)
    if 'x' in pattern_position_tl.keys():
        plt.scatter(pattern_position_tl['x'], pattern_position_tl['y'], c='magenta', marker='>')
    plt.gca().set_title(f"Mouse Position: ({x_position}, {y_position})")
    plt.draw()


def onmove(event, img):
    """
    Mouse moving on figure event callback function
    """
    global x_position, y_position
    if event.xdata is not None and event.ydata is not None:
        x_position = int(event.xdata)
        y_position = int(event.ydata)
        update_image(img)


def onclick_tl(event):
    """
    Top left pattern corner mark on mouse click event callback function
    """
    if event.button == 1:  # Left mouse button clicked
        pattern_position_tl['x'] = int(event.xdata)
        pattern_position_tl['y'] = int(event.ydata)
        plt.scatter(pattern_position_tl['x'], pattern_position_tl['y'], c='magenta', marker='>')
        plt.pause(0.1)
        plt.close()


def onclick_br(event):
    """
    Bottom right pattern corner mark on mouse click event callback function
    """
    if event.button == 1:  # Left mouse button clicked
        pattern_position_br['x'] = int(event.xdata)
        pattern_position_br['y'] = int(event.ydata)
        plt.scatter(pattern_position_br['x'], pattern_position_br['y'], c='magenta', marker='<')
        plt.pause(0.1)
        plt.close()


def find_calibration_pattern(image_a, image_b, pattern_size):
    """
    :param image_a: First pattern image
    :param image_b: Second pattern image
    :param pattern_size: number of corners in pattern
    :return: lists of 3D world coordinates obj,  pattern corners for image a,  pattern corners for image b
    """
    global pattern_position_tl
    global pattern_position_br
    obj_points = []  # 3D world coordinates
    img_points = []  # Image pattern corners
    for e, img in enumerate([image_a, image_b]):
        pattern_position_tl = {}
        pattern_position_br = {}
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fig = plt.figure()
        plt.imshow(img_rgb)
        plt.title('Top left of pattern')
        fig.canvas.mpl_connect("button_press_event", onclick_tl)
        fig.canvas.mpl_connect("motion_notify_event", lambda event: onmove(event, img_rgb))
        plt.show()
        fig = plt.figure()
        plt.imshow(img_rgb)
        plt.scatter(pattern_position_tl['x'], pattern_position_tl['y'], c='magenta', marker='>')
        plt.title('Bottom right of pattern')
        fig.canvas.mpl_connect("button_press_event", onclick_br)
        fig.canvas.mpl_connect("motion_notify_event", lambda event: onmove(event, img_rgb))
        plt.show()
        # Find the calibration pattern (checkerboard, circles, etc.)
        print(pattern_position_br)
        print(pattern_position_tl)
        focus_pattern = cv2.adaptiveThreshold(gray[pattern_position_tl['y']: pattern_position_br['y'],
                                              pattern_position_tl['x']: pattern_position_br['x']], 255,
                                              cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 125, 10)
        plt.imshow(focus_pattern, cmap='gray')
        plt.show()
        ret, relative_corners = cv2.findChessboardCorners(focus_pattern, pattern_size, None)
        corners = np.array(
            [[[rc[0][0] + pattern_position_tl['x'], rc[0][1] + pattern_position_tl['y']]] for rc in relative_corners])
        plt.imshow(gray, cmap='gray')
        for corner in corners:
            plt.scatter(corner[0][0], corner[0][1], c='r', s=5)
        plt.show()

        if ret:
            # Append the object points (same for all images)
            objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
            obj_points.append(objp)

            # Refine the corner coordinates
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners_refine = cv2.cornerSubPix(gray, corners.astype(np.float32), (11, 11), (-1, -1), criteria)

            img_points.append(corners_refine)
    return obj_points, img_points


def drawMatches(imageA, imageB, kpsA, kpsB):
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB
    # draw the match
    for e, (ptA, ptB) in enumerate(zip(kpsA, kpsB)):
        ptA_line = (int(ptA[0][0]), int(ptA[0][1]))
        ptB_line = (int(ptB[0][0]) + wA, int(ptB[0][1]))
        cv2.line(vis, ptA_line, ptB_line, (np.random.randint(0,255), np.random.randint(0,255),
                                           np.random.randint(0,255)), 3)
    # return the visualization
    plt.imshow(vis)
    plt.show()


def calibrate_2_image(image_a_path, image_b_path, pattern_size):
    """
    :param image_a_path: Path of first image for calibration
    :param image_b_path: Path of first image for calibration
    :param pattern_size: Number of corners in pattern
    :return:
    """
    img_a = cv2.imread(image_a_path)
    img_b = cv2.imread(image_b_path)
    obj_points, img_points = find_calibration_pattern(img_a, img_b, pattern_size)
    # Calibrate cameras
    img_points1_sorted = [np.array([point for point in sorted(img_points[0], key=lambda x: x[0][0])])[6:]]
    img_points2_sorted = [np.array([point for point in sorted(img_points[1], key=lambda x: x[0][0])])[6:]]
    obj_points[0] = obj_points[0][6:]
    obj_points[1] = obj_points[1][6:]
    drawMatches(img_a, img_b, img_points1_sorted[0], img_points2_sorted[0])
    _, mtx1, dist1, _, _ = cv2.calibrateCamera([obj_points[0]], img_points1_sorted, img_a[:, :, 0].shape[::-1],
                                               None, None)
    _, mtx2, dist2, _, _ = cv2.calibrateCamera([obj_points[1]], img_points2_sorted, img_b[:, :, 0].shape[::-1],
                                               None, None)

    # Stereo Calibration
    retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate([obj_points[0]], img_points1_sorted,
                                                         img_points2_sorted, mtx1, dist1, mtx2, dist2,
                                                         img_a[:, :, 0].shape[::-1])

    # Rectify Images
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, img_a[:, :, 0].shape[::-1], R, T)
    map1x, map1y = cv2.initUndistortRectifyMap(mtx1, dist1, R1, P1, img_a[:, :, 0].shape[::-1], cv2.CV_32F)
    map2x, map2y = cv2.initUndistortRectifyMap(mtx2, dist2, R2, P2, img_a[:, :, 0].shape[::-1], cv2.CV_32F)

    # Remape the images
    rectified_img1 = cv2.remap(img_a, map1x, map1y, cv2.INTER_LINEAR)
    rectified_img2 = cv2.remap(img_b, map2x, map2y, cv2.INTER_LINEAR)

    plt.close()
    plt.subplot(121)
    plt.imshow(rectified_img1)
    plt.subplot(122)
    plt.imshow(rectified_img2)
    plt.show()

    return rectified_img1, rectified_img2
