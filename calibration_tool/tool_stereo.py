import argparse
import os
import glob
import cv2
import numpy as np
import calib
import pickle

from threaded_capture import ThreadedCapture
import stereo_camera


def parse_checkerboard(checkerboard):
    return tuple(map(int, checkerboard.split("x")))


def parse_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "-v", "--video",
        default=stereo_camera.DEFAULT_VIDEO_DEVICE,
        help="video device to be opened for calibration eg. 0"
    )

    arg_parser.add_argument(
        "-r", "--resolution",
        default=stereo_camera.DEFAULT_RESOLUTION,
        help=f"requested resolution. supported are: {', '.join(stereo_camera.SUPPORTED_RESOLUTIONS.keys())} "
    )

    arg_parser.add_argument(
        "-c", "--checkerboard",
        default='10x7',
        help="checkerboard eg. '10x7'"
    )

    arg_parser.add_argument(
        "-sq", "--square-size",
        type=float,
        default=24.2,
        help="checkerboard square size eg. 24.2"
    )

    arg_parser.add_argument(
        "-d", "--data",
        default="data",
        help="data directory where to store calibration images, captures and results"
    )

    args = arg_parser.parse_args()

    # parse checkerboard
    args.checkerboard = parse_checkerboard(args.checkerboard)
    args.resolution = stereo_camera.SUPPORTED_RESOLUTIONS[args.resolution]
    args.video = int(args.video)
    args.square_size = float(args.square_size)
    return args


def save_capture(
        args,
        image,
        number,
        cam_name,
        name
):
    file_name = os.path.join(
        args.data,
        name,
        f"{cam_name}_{number}.jpg"
    )
    print(f"saving screenshot {file_name}")
    cv2.imwrite(
        file_name, image
    )


def detect_checkerboard(args, image):
    checkerboard = args.checkerboard

    # 1. convert image to grayscale
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = calib.detect_board(
        checkerboard,
        gray
    )

    detection = {
        "found": found,
        "corners": corners,
        "image": image.copy()
    }

    return detection


def add_detection(
        args,
        object_points,
        calibration,
        detection_left,
        detection_right,
        save=True
):
    if detection_left is None or detection_right is None:
        print("please enable detection using 'd' key")

    elif not detection_left['found'] or not detection_right['found']:

        print("board was not found, cannot add image to set")

    else:

        img_number = calibration["image_number"]

        left_image = detection_left["image"]
        left_corners = detection_left["corners"]
        w, h = left_image.shape[1], left_image.shape[0]

        calibration["left"]["detections"].append(detection_left)
        calibration["left"]["object_points"].append(object_points)
        calibration["left"]["image_points"].append(left_corners)

        if calibration["left"]["image_shape"] is None:
            calibration["left"]["image_shape"] = (w, h)
        else:
            if calibration["left"]["image_shape"] != (w, h):
                print("warning, image shape has changed!")

        if save:
            save_capture(args, left_image, img_number, "left", "calib")

        right_image = detection_right["image"]
        right_corners = detection_right["corners"]
        w, h = right_image.shape[1], right_image.shape[0]

        calibration["right"]["detections"].append(detection_right)
        calibration["right"]["object_points"].append(object_points)
        calibration["right"]["image_points"].append(right_corners)

        if calibration["right"]["image_shape"] is None:
            calibration["right"]["image_shape"] = (w, h)
        else:
            if calibration["right"]["image_shape"] != (w, h):
                print("warning, image shape has changed!")

        if save:
            save_capture(args, right_image, img_number, "right", "calib")

        calibration["image_number"] += 1


def np_print(np_array):
    h, w = np_array.shape
    if h == 1 or w == 1:
        num_fmt = "{:.6f}"
    else:
        num_fmt = "{:.3f}"

    str_array = "[\n" + ",\n".join([
        "\t[" + ",\t".join([num_fmt.format(v).rjust(10, ' ') for v in row]) + "]"
        for row in np_array
    ]) + "\n]"
    ret = "np.array(" + str_array + ")"
    return ret


def calibrate(calibration):
    obj_points = calibration["object_points"]
    world_points = calibration["image_points"]
    img_shape = calibration["image_shape"]

    print("num_points", len(obj_points))
    print("calibrating...")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        world_points,
        img_shape, None, None
    )

    # np.set_printoptions(suppress=True)
    # print("Camera matrix : \n")
    # cam_matrix = mtx.round(3)
    # print([list(i) for i in cam_matrix])
    ## print(mtx.round(3))
    #print("dist : \n")
    # print(dist)
    print("# Intrinsic Camera Parameters")
    print("cam_matrix = " + np_print(K))

    print("# Distortion Coefficients")
    print("dist_coeffs = " + np_print(dist))

    return K, dist, rvecs, tvecs


def calibrate_stereo(args, calibration_info, left_calibration, right_calibration):

    left_info = calibration_info["left"]
    right_info = calibration_info["right"]

    left_images_points = left_info["image_points"]
    right_images_points = right_info["image_points"]
    world_points = left_info["object_points"]
    image_size = left_info["image_shape"]

    world_points = [p.reshape(-1, 3) for p in world_points]
    left_images_points = [p.reshape(-1, 2) for p in left_images_points]
    right_images_points = [p.reshape(-1, 2) for p in right_images_points]

    left_K, left_dist = left_calibration[0], left_calibration[1]
    right_K, right_dist = right_calibration[0], right_calibration[1]
    err, left_K, left_dist, right_K, right_dist, R, T, E, F = cv2.stereoCalibrate(
        world_points,
        left_images_points,
        right_images_points,
        left_K,
        left_dist,
        right_K,
        right_dist,
        image_size,
        flags=0
    )

    to_print = [

        "# Left camera Intrinsics:",
        ("left_K", left_K),
        ("left_dist", left_dist),

        "# Right camera Intrinsics:",
        ("right_K", right_K),
        ("right_dist", right_dist),

        "# Rotation:",
        ("R", R),

        "# Translation:",
        ("T", T),

        "# Essential Matrix:",
        ("E", E),

        "# Fundamental Matrix:",
        ("F", F),

    ]
    print("# STEREO CALIBRATION")
    for line in to_print:

        if isinstance(line, str):
            print(line)
        else:
            var_name, np_array = line
            print(f"{var_name} = {np_print(np_array)}\n")

    calibration_results = {
        'left_K': left_K,
        'left_dist': left_dist,
        'right_K': right_K,
        'right_dist': right_dist,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        'image_size': image_size,
        # 'left_pts': left_pts,
        # 'right_pts': right_pts
    }

    calibration_file = os.path.join(args.data, "stereo_calibration.pkl")
    with open(calibration_file, "wb") as f:
        f.write(pickle.dumps(calibration_results))

    return calibration_results


def create_stereo_rectifying_maps(calibration_results):
    left_K = calibration_results['left_K']
    left_dist = calibration_results['left_dist']
    right_K = calibration_results['right_K']
    right_dist = calibration_results['right_dist']
    image_size = calibration_results['image_size']
    R = calibration_results['R']
    T = calibration_results['T']

    print("rectifying stereo...")
    R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(
        left_K, left_dist, right_K, right_dist, image_size, R, T, alpha=0
    )

    print("creating undistortion maps...")
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(left_K, left_dist, R1, P1, image_size, cv2.CV_32FC1)
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(right_K, right_dist, R2, P2, image_size, cv2.CV_32FC1)

    stereo_maps = {

        # undistorting maps
        "left_map_x": left_map_x,
        "left_map_y": left_map_y,
        "right_map_x": right_map_x,
        "right_map_y": right_map_y,

        # add also rectifying info:
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "validRoi1": validRoi1,
        "validRoi2": validRoi2,

    }

    stereo_maps_file = os.path.join(args.data, "stereo_maps.pkl")
    with open(stereo_maps_file, "wb") as f:
        f.write(pickle.dumps(stereo_maps))

    return stereo_maps


def new_calibration():
    calibration = {
        "image_number": 0,

        "left": {
            "image_shape": None,
            "detections": [],
            "object_points": [],  # board points
            "image_points": [],  # locations found in the image
        },
        "right": {
            "image_shape": None,
            "detections": [],
            "object_points": [],  # board points
            "image_points": [],  # locations found in the image
        }

    }
    return calibration


def read_calibration(
        args
):
    checkerboard = args.checkerboard
    checkerboard_world_points = args.square_size * calib.board_points(checkerboard)

    calibration = new_calibration()

    directory = os.path.join(args.data, "calib")
    left_files_pattern = "*left*.jpg"
    right_files_pattern = "*right*.jpg"

    def numeric_sort(file_name):
        return int(file_name.split("_")[-1].split(".")[0])

    left_file_names = sorted(
        glob.glob(
            os.path.join(directory, left_files_pattern)
        ),
        key=numeric_sort
    )

    right_file_names = sorted(
        glob.glob(
            os.path.join(directory, right_files_pattern)
        ),
        key=numeric_sort
    )

    num_left = len(left_file_names)
    num_right = len(right_file_names)

    if num_left != num_right:
        raise Exception(f"the number of files (left {num_left} / right{num_right}) doesn't match")
    image_size = None

    for left_file_name, right_file_name in zip(
            left_file_names, right_file_names
    ):

        print("processing", left_file_name, right_file_name)

        # read left and right images
        left_image = cv2.imread(left_file_name, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(right_file_name, cv2.IMREAD_GRAYSCALE)

        # get the images sizes
        left_size = (left_image.shape[1], left_image.shape[0])
        right_size = (right_image.shape[1], right_image.shape[0])

        # checks that images sizes match
        if left_size != right_size:
            raise Exception(f"left and right images sizes differ: left {left_size} / right {right_size}")

        if image_size is None:
            # remembers the images size
            image_size = left_size
        else:
            if image_size != left_size:
                raise Exception(f"there are images with different sizes: {image_size} vs {left_size}")

        # finds the checkerboard in each image
        detection_left = detect_checkerboard(args, left_image)
        detection_right = detect_checkerboard(args, right_image)

        if not detection_left["found"] or not detection_right["found"]:
            print("warning, checkerboard was not found")
            continue

        # checkerboard was found in both images.

        add_detection(
            args,
            checkerboard_world_points,
            calibration,
            detection_left,
            detection_right,
            save=False
        )
    return calibration


def make_dirs(args):

    data_dir = args.data
    calib_dir = os.path.join(data_dir, "calib")
    captures_dir = os.path.join(data_dir, "captures")

    for d in [data_dir, captures_dir, calib_dir]:
        if not os.path.exists(d):
            print(f"Directory {d} doesn't exist, creating...")
            os.makedirs(d)


def start(args):
    make_dirs(args)

    resolution = args.resolution
    device = args.video

    cap = stereo_camera.new_video_capture(device, resolution)

    if not cap.isOpened():
        print("Camera could not be opened, you can load calibration images using the l key")
        #exit()

    checkerboard = args.checkerboard
    checkerboard_world_points = args.square_size * calib.board_points(checkerboard)
    detection_enabled = False
    detection_left = None
    detection_right = None

    calibration = new_calibration()

    draw_corners = True
    capture_no = 0
    frame_no = 0

    calibration_results = None
    if cap.isOpened():
        th_cap = ThreadedCapture(cap)
        th_cap.start()

    window_flags = cv2.WINDOW_NORMAL
    # window_flags = cv2.WINDOW_FREERATIO
    cv2.namedWindow("stereo", window_flags)

    while True:

        if cap.isOpened():
            # Capture frame-by-frame
            ret, frame = th_cap.read()
            # ret, frame = cap.read()

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame_no += 1
            if frame_no == 1:
                print("resolution: ", (frame.shape[1], frame.shape[0]))
            #if frame_no % 2 == 0:
            #    continue

            # split frame
            shape = frame.shape
            w, h = shape[1], shape[0]

            is_stereo = resolution == (w, h)
            if not is_stereo:
                message = f"specified camara ({args.video}) IS NOT the stereo camera"
                print("terminating.", message)
                break

            left_frame = frame[:, :int(w / 2), :]
            right_frame = frame[:, int(w / 2):, :]

            # left_frame = cv2.rotate(left_frame, cv2.ROTATE_180)
            # right_frame = cv2.rotate(right_frame, cv2.ROTATE_180)

            show_img_left = left_frame.copy()
            show_img_right = right_frame.copy()

            # draws so-far detected corners
            if draw_corners:

                for corners in calibration["left"]["image_points"]:
                    show_img_left = cv2.drawChessboardCorners(
                        show_img_left, args.checkerboard, corners, True
                    )

                for corners in calibration["right"]["image_points"]:
                    show_img_right = cv2.drawChessboardCorners(
                        show_img_right, args.checkerboard, corners, True
                    )

            if detection_enabled:

                # detects board
                detection_left = detect_checkerboard(args, left_frame)
                detection_right = detect_checkerboard(args, right_frame)

                found = detection_left['found']
                if found:
                    show_img_left = calib.draw_checkerboard(
                        show_img_left,
                        args.checkerboard,
                        detection_left['corners'],
                        found
                    )

                found = detection_right['found']
                if found:
                    show_img_right = calib.draw_checkerboard(
                        show_img_right,
                        args.checkerboard,
                        detection_right['corners'],
                        found,
                    )

            #cv2.imshow('left', show_img_left)
            #cv2.imshow('right', show_img_right)
            show_img = np.hstack((show_img_left, show_img_right))
            show_img = cv2.resize(show_img, (int(w / 2), int(h / 2)))
            cv2.imshow("stereo", show_img)

        k = cv2.waitKey(10)

        if k == ord('h'):

            keys_help = """
            
                h: help, 
                    Muestra ayuda.
                    
                q: quit, 
                    Termina la app
                    
                d: detect, 
                    Habilita o inhabilita la detección del checkerboard.
                    
                a: add, 
                    Agrega detecciones del checkerboard al set de calibración.
                    Debe estar la detección habilitada.
                    
                c: calibrate, 
                    Realiza la calibración estéreo.
                    Deben haberse detectado al menos 10 pares estéreo de checkerboards.
                    Guarda los resultados de calibración en un archivo pickle stereo_calibration.pkl

                m: rectification maps, 
                    Crea los mapas de rectification tanto de corrección de distorsión para cada lente,
                    cómo los mapas de rectificación estéreo.
                    Guarda los mapas de rectificación en el archivo pickle stereo_calibration.pkl
                    
                s: snapshot, 
                    Captura par estéreo de imágenes (left, right)
                    y las guarda en el directorio de capturas.
                    
                l: load calibration images,
                    Lee las imágenes del directorio de calibración, 
                    y reprocesa todas detecciones de checkerboards.
                    
            
            """
            print(keys_help)

        elif k == ord('q'):
            # quit
            break

        elif k == ord('d'):

            # toggles detection on / off
            detection_enabled = not detection_enabled
            if not detection_enabled:
                detection_left = None
                detection_right = None

        elif k == ord('a'):

            # add image to calibration set

            add_detection(
                args,
                checkerboard_world_points,
                calibration,
                detection_left,
                detection_right
            )

        elif k == ord('c'):

            if len(calibration["left"]["detections"]) < 10:
                print("not enough images to calibrate")
            else:

                # calibrate
                print("LEFT CALIBRATION:")
                calib_left = calibrate(calibration["left"])

                print("RIGHT CALIBRATION:")
                calib_right = calibrate(calibration["right"])

                print("STEREO CALIBRATION:")
                calibration_results = calibrate_stereo(args, calibration, calib_left, calib_right)

        elif k == ord('m'):

            if calibration_results is None:
                print("can't create rectification maps, first calibrate stereo.")
            else:
                print(f"creating rectification maps...")
                create_stereo_rectifying_maps(calibration_results)

        elif k == ord('s'):

            print(f"saving snapshot: {capture_no}")
            save_capture(args, left_frame, capture_no, "left", "captures")
            save_capture(args, right_frame, capture_no, "right", "captures")
            capture_no += 1

        elif k == ord('l'):

            print(f"loading calibration images:")
            calibration = read_calibration(args)
            print("calibration done")

    # When everything done, release the capture
    if cap.isOpened():
        th_cap.stop()
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()

    start(args)
