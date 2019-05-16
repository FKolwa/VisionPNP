import VisionPNP
import cv2
import numpy as np

# position = VisionPNP.findShape('./images/tray_resistor.png')
# print(position)

maskValues = VisionPNP.getHSVColorRange('./images/gripper.png')
rawImage = cv2.imread('./images/tray_resistor.png')

maskImage = VisionPNP.createColorRangeMask(rawImage, maskValues)
maskImageConv = np.array(maskImage)
cv2.imwrite('./01_mask.png', maskImageConv)

croppedImage = VisionPNP.cropImageToMask(rawImage, maskImageConv)
croppedImage = np.array(croppedImage, copy=False)
cv2.imwrite('./02_cropped.png', croppedImage)

# inputImg = cv2.imread('./images/tray_resistor.png')
# transfered = VisionPNP.returnImage(inputImg)
# returnedImg = np.array(transfered, copy=False)
# cv2.imwrite('./returned-result.png', returnedImg)

# maskValues = VisionPNP.getHSVColorRange('./images/gripper.png')
# print(maskValues)

# imageType = cv2.imread('./images/tiny-on-gripper.png')
# maskedImage = VisionPNP.removeColorRange(imageType, maskValues)

# orientation = VisionPNP.matchTemplate('./images/tiny-on-gripper.png', './images/template-output.png', maskValues)
# print(orientation)



