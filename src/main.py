import argparse
from calibrate_2_images import calibrate_2_image
from create_3d_image import reconstruct_3d
import cv2



def get_args():
    # Define user input arguments
    parser = argparse.ArgumentParser(prog='3d_from_2_cameras', description='Calibrate 2 cameras and create 3d image')
    parser.add_argument('-a', '--image_a', type=str, required=True,
                        help='First image for calibration path')
    parser.add_argument('-b', '--image_b', type=str, required=True,
                        help='Second image for calibration path')
    parser.add_argument('-f', '--focal_length', type=float, required=True,
                        help='focal length of camera in pixels')
    parser.add_argument('-B', '--baseline_distance', type=float, required=True,
                        help='base line distance between cameras in meters')
    parser.add_argument('--pattern_size', '--pattern_size', nargs='+', type=int, default=None,
                        help='Size of calibration pattern (number of corners)')
    parser.add_argument('--calibrate', '--calibrate', type=bool, default=True,
                        help='True if requires calibration between cameras')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.calibrate and args.pattern_size is None:
        print('pattern_size is required for calibration!')
        exit()
    elif args.calibrate:
        rectified_img1, rectified_img2 = calibrate_2_image(args.image_a, args.image_b, args.pattern_size)
        reconstruct_3d(rectified_img1, rectified_img2, f=args.focal_length, B=args.baseline_distance)
    else:
        reconstruct_3d(cv2.imread(args.image_a), cv2.imread(args.image_b),
                       f=args.focal_length, B=args.baseline_distance)


