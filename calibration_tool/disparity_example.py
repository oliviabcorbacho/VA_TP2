import cv2
import json
import numpy as np
from pathlib import Path

from ..stereodemo.method_cre_stereo import CREStereo
from ..stereodemo.method_opencv_bm import StereoBM, StereoSGBM
from ..stereodemo.methods import Calibration, InputPair, Config




def create_disparity_map(left_image, right_image, pair_number, calibration_file, method_name, save=False, show=False):

    models_path = Path.home() / ".cache" / "stereodemo" / "models"

    print("Computing disparity...")

    with open(calibration_file, "r") as f:
        calibration = Calibration.from_json(f.read())

    pair = InputPair(left_image, right_image, calibration, "status?")
    config = Config(models_path=models_path)

    if method_name== "CREStereo":
        method = CREStereo(config)
    elif method_name == "StereoBM":
        method = StereoBM(config)
    elif method_name == "StereoSGBM":
        method = StereoSGBM(config)

    disparity = method.compute_disparity(pair)

    if save:
        np.savez(f'../mapas_disparidad/disparity_{pair_number}.npz', disparity.disparity_pixels)
        cv2.imwrite(f"../mapas_disparidad/disparity_{pair_number}.png", disparity.disparity_pixels)

    if show:
        dvis = disparity.disparity_pixels.copy()
        dvis = 255 * (dvis - dvis.min()) / (dvis.max() - dvis.min())
        dvis = dvis.astype('uint8')
        cv2.imshow("disparity", dvis)

        input = np.hstack((left_image, right_image))
        cv2.imshow("input", input)
        cv2.waitKey()

    return

