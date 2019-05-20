import VisionPNP
import cv2
import numpy as np

#---------<
# Input images
headCamImage = cv2.imread('./images/tray_resistor.png')
bedCamImage = cv2.imread('./images/tiny-on-gripper.png')
templateImage = cv2.imread('./images/template-output.png')


#---------<
# Scenario 1 - Find position of object in tray picture
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
cv2.imwrite('./01_object_com_tray.png', croppedImageConv)

#---------<
# Scenario 2 - Find position of object in gripper image (no orientation)
#---------<
# Create working copy
bedCamImageCopy = bedCamImage.copy()
# Clean green background
cleanedBedCamRaw = VisionPNP.removeColorRange(bedCamImageCopy, maskValues)
cleanedBedCam = np.array(cleanedBedCamRaw)

# Find center of mass
center = VisionPNP.findShape(cleanedBedCam)
print(center)
cv2.circle(bedCamImageCopy,(center[0], center[1]), 4, (0,0,255), -1)
cv2.imwrite('./02_object_com_gripper.png', bedCamImageCopy)


#---------<
# Scenario 3 - Find template in search image (with orientation)
#---------<
# Find a binary template inside a provided input image.
# The maskValues contain the color range of the background color for easier seperation.
# Return its orientation.
orientation = VisionPNP.matchTemplate('./images/tiny-on-gripper.png', './images/template-output.png', maskValues)
print(orientation)
