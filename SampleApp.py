import VisionPNP
import cv2
import numpy as np


# position = VisionPNP.findShape('./images/tray_resistor.png')
# print(position)

maskValues = VisionPNP.getHSVColorRange('./images/gripper.png')
rawImage = cv2.imread('./images/tray_resistor.png')

maskImage = VisionPNP.createColorRangeMask(rawImage, ((30, 45, 45), (90, 255, 255)))
maskImageConv = np.array(maskImage)
cv2.imwrite('./01_mask.png', maskImageConv)

croppedImage = VisionPNP.cropImageToMask(rawImage, maskImageConv)
croppedImageConv = np.array(croppedImage, copy=False)
cv2.imwrite('./02_cropped.png', croppedImageConv)

# maskValues = VisionPNP.getHSVColorRange('./images/gripper.png')
# print(maskValues)

# imageType = cv2.imread('./images/tiny-on-gripper.png')
# maskedImage = VisionPNP.removeColorRange(imageType, maskValues)

# orientation = VisionPNP.matchTemplate('./images/tiny-on-gripper.png', './images/template-output.png', maskValues)
# print(orientation)
