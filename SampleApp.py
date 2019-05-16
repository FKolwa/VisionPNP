import VisionPNP
import cv2
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

# position = VisionPNP.findShape('./images/tray_resistor.png')
# print(position)

maskValues = VisionPNP.getHSVColorRange('./images/gripper.png')
rawImage = cv2.imread('./images/tray_resistor.png')

maskImage = VisionPNP.createColorRangeMask(rawImage, maskValues)
maskImageConv = np.array(maskImage)
result = maskImageConv[:, :, 0]
cv2.imwrite('./01_mask.png', result)

croppedImage = VisionPNP.cropImageToMask(rawImage, result)
croppedImageConv = np.array(croppedImage, copy=False)
print("Cropped")
print(croppedImageConv.shape)
cv2.imwrite('./02_cropped.png', croppedImageConv)

# maskValues = VisionPNP.getHSVColorRange('./images/gripper.png')
# print(maskValues)

# imageType = cv2.imread('./images/tiny-on-gripper.png')
# maskedImage = VisionPNP.removeColorRange(imageType, maskValues)

# orientation = VisionPNP.matchTemplate('./images/tiny-on-gripper.png', './images/template-output.png', maskValues)
# print(orientation)
