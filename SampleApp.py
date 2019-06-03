import VisionPNP
import cv2
import numpy as np

#---------<
# Input images
headCamImage = cv2.imread('./resources/tiny_on_tray.png')
bedCamImage = cv2.imread('./resources/resistor_on_gripper.png')
templateImage = cv2.imread('./resources/template_output_resistor.png', cv2.IMREAD_GRAYSCALE)

#---------<
# Scenario 1 - Find position of object in tray picture
#---------<
# Read the HSV color range from a background image
maskValues = VisionPNP.getHSVColorRange('./resources/gripper.png')

# Create a binarized image of th input containing only the areas within the
# provided color mask (black)
maskImage = VisionPNP.createColorRangeMask(headCamImage, maskValues)

# Crop the input image to the shape of the provided mask
croppedImage = VisionPNP.cropImageToMask(headCamImage, maskImage)
cv2.imwrite('./01_cropped_image_ony_tray.png', croppedImage)

# Find the position of a single object inside the provided image
position = VisionPNP.findShape(croppedImage)
cv2.circle(croppedImage,(position[0], position[1]), 4, (0,0,255), -1)
cv2.imwrite('./01_object_on_tray.png', croppedImage)
print(position)

#---------<
# Scenario 1B - Extract bouding rect from mask, then use rect to crop image.
#---------<
# Extract bouding rect fom binarized mask image
bRect = VisionPNP.findContainedRect(maskImage)
croppedImage1B = VisionPNP.cropImageToRect(headCamImage, bRect)

cv2.imwrite('./01B_tray_image_cropped.png', croppedImage1B)

#---------<
# Scenario 2 - Find position of object in gripper image (no orientation)
#---------<
# Create working copy
bedCamImageCopy = bedCamImage.copy()

# Clean green background
cleanedBedCam = VisionPNP.removeColorRange(bedCamImageCopy, maskValues)

# Find center of mass
center = VisionPNP.findShape(cleanedBedCam)
cv2.circle(bedCamImageCopy,(center[0], center[1]), 4, (0,0,255), -1)
cv2.imwrite('./02_object_on_gripper.png', bedCamImageCopy)
print(center)

#---------<
# Scenario 3 - Find template in search image (with orientation)
#---------<
# Find a binary template inside a provided input image.
# The maskValues contain the color range of the background color for easier seperation.
# Return its orientation.
orientation = VisionPNP.matchTemplate(bedCamImage, templateImage, maskValues, './resources/houghConfig.json')
print(orientation)
