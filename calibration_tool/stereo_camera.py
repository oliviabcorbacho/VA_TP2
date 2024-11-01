import cv2
import numpy as np
from . import utils
import platform
import time

from .threaded_capture import ThreadedCapture

DEFAULT_VIDEO_DEVICE = 2
# DEFAULT_VIDEO_DEVICE = "/dev/video3"

# Resolución a usar en cada cámara:
DEFAULT_RESOLUTION = '1920x1080'
# DEFAULT_RESOLUTION = '800x600'

FPS = 30

OS_PLATFORM = platform.platform()
OS_SYSTEM = platform.system()

VIDEO_CAPTURE_MODE = cv2.CAP_ANY
if OS_SYSTEM == 'Windows':
    VIDEO_CAPTURE_MODE = cv2.CAP_DSHOW


SUPPORTED_RESOLUTIONS = {
    'default': None,
    '1920x1080': (3840, 1080),  # each cam: full hd (1290x1080)
    '1280x720': (2560, 720),  # each cam: (1280x720)
    '800x600': (1600, 600),  # each cam: (800x600)
    '640x480': (1280, 480),  # each cam: (640x480)
    '320x240': (640, 240),  # each cam: (320x240)
}

# Dispositivo de captura.
# depende cómo lo maneja el sistema operativo.
# - puede ser un número, en ese caso probamos 0, 1, 2...
# - en linux es un archivo que está en /dev/video*


def new_video_capture(device, resolution):
    print(device, " ", resolution)
    cap = cv2.VideoCapture(device, VIDEO_CAPTURE_MODE)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if resolution is not None:
        req_w, req_h = resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)

    return cap


def start():
    # starts capture
    resolution = SUPPORTED_RESOLUTIONS[DEFAULT_RESOLUTION]
    cap = new_video_capture(DEFAULT_VIDEO_DEVICE, resolution)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    window_flags = cv2.WINDOW_NORMAL
    # window_flags = cv2.WINDOW_FREERATIO
    cv2.namedWindow("stereo", window_flags)

    frame_no = 0

    th_cap = ThreadedCapture(cap)
    th_cap.start()

    while True:

        # Capture frame-by-frame
        # ret, frame = cap.read()
        ret, frame = th_cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_no += 1
        if frame_no == 1:
            print("resolution: ", (frame.shape[1], frame.shape[0]))

        # split frame
        shape = frame.shape
        w, h = shape[1], shape[0]
        left_frame = frame[:, :int(w / 2), :]
        right_frame = frame[:, int(w / 2):, :]

        # left_frame = cv2.rotate(left_frame, cv2.ROTATE_180)
        # right_frame = cv2.rotate(right_frame, cv2.ROTATE_180)

        show_img_left = left_frame.copy()
        show_img_right = right_frame.copy()

        is_stereo = resolution == (w, h)
        if is_stereo:
            utils.draw_text(show_img_left, "left", (20, 20), font_scale=8, font_thickness=3)
            utils.draw_text(show_img_right, "right", (20, 20), font_scale=8, font_thickness=3)

        # cv2.imshow('left', show_img_left)
        # cv2.imshow('right', show_img_right)

        show_img = np.hstack((show_img_left, show_img_right))
        if not is_stereo:
            utils.draw_text(show_img, "not stereo", (20, 20), font_scale=8, font_thickness=3, text_color=(0, 0, 255))
        cv2.imshow("stereo", show_img)

        k = cv2.waitKey(1)
        if k == ord('q'):
            # quit
            break

    # When everything done, release the capture
    th_cap.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    start()
