import VisionPNP
import cv2
import numpy as np

#---------<
# Input images
headCamImage = cv2.imread('./images/tray_resistor.png')
bedCamImage = cv2.imread('./images/tiny-on-gripper.png')
templateImage = cv2.imread('./images/template-output.png')

#---------<
# Read the HSV color range from a background image
maskValues = VisionPNP.getHSVColorRange('./images/gripper.png')

#---------<
# Create a binarized image of th input containing only the areas within the
# provided color mask (black)
maskImage = VisionPNP.createColorRangeMask(headCamImage, maskValues)
maskImageConv = np.array(maskImage)

#---------<
# Crop the input image to the shape of the provided mask
croppedImage = VisionPNP.cropImageToMask(headCamImage, maskImageConv)
croppedImageConv = np.array(croppedImage)

#---------<
# Find the position of a single object inside the provided image
position = VisionPNP.findShape(croppedImageConv)
print(position)
cv2.circle(croppedImageConv,(position[0], position[1]), 4, (0,0,255), -1)
cv2.imwrite('./01_object_position.png', croppedImageConv)

#---------<
# Find a binary template inside a provided input image.
# The maskValues contain the color range of the background color for easier seperation.
# Return its orientation.
orientation = VisionPNP.matchTemplate('./images/tiny-on-gripper.png', './images/template-output.png', maskValues)
print(orientation)
