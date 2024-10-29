import cv2
import json
import numpy as np
from pathlib import Path

from stereodemo.method_cre_stereo import CREStereo
print("Importing modules...")
from stereodemo.method_opencv_bm import StereoBM, StereoSGBM
from stereodemo.methods import Calibration, InputPair, Config





models_path = Path.home() / ".cache" / "stereodemo" / "models"


calibration_file = "/Users/brunocr/Desktop/UDESA/Visión Artificial/TP2_Reconstruccion3D/VA_TP2/calibration_tool/data/rectified_images/stereodemo_calibration.json"

print("Computing disparity...")

with open(calibration_file, "r") as f:
    calibration = Calibration.from_json(f.read())

left_image = cv2.imread("/Users/brunocr/Desktop/UDESA/Visión Artificial/TP2_Reconstruccion3D/VA_TP2/calibration_tool/data/rectified_images/rectified_left_5.png")
right_image = cv2.imread("/Users/brunocr/Desktop/UDESA/Visión Artificial/TP2_Reconstruccion3D/VA_TP2/calibration_tool/data/rectified_images/rectified_right_5.png")

pair = InputPair(left_image, right_image, calibration, "status?")
config = Config(models_path=models_path)



method = CREStereo(config)
# method = StereoBM(config)
# medhod = StereoSGBM(config)
disparity = method.compute_disparity(pair)


# np.savez("disparity.npz", disparity.disparity_pixels)
# cv2.imwrite("disparity.png", disparity.disparity_pixels)

dvis = disparity.disparity_pixels.copy()
dvis = 255 * (dvis - dvis.min()) / (dvis.max() - dvis.min())
dvis = dvis.astype('uint8')
cv2.imshow("disparity", dvis)

input = np.hstack((left_image, right_image))
cv2.imshow("input", input)
cv2.waitKey()