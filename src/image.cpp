#include "../include/image.h"

//--------------------------------------------------------<
// Match template in image and return rotation
float Image::matchTemplate(const cv::Mat& inputImage,const cv::Mat& templateImage, const std::vector<std::vector<int>>& colorRange) {
  bool DEBUG = true;
  std::vector<cv::Vec4f> detected;
  cv::TickMeter tm;
  cv::Mat outputImage = inputImage.clone();
  cv::resize(outputImage, outputImage, cv::Size(), 0.25, 0.25);
  cv::Mat workImage;
  cv::Mat templateScaled = templateImage.clone();

  // Remove all the color within the color range
  workImage = removeColorRange(inputImage, colorRange);

  // Preprocess raw image data
  cv::cvtColor(workImage, workImage, cv::COLOR_BGR2GRAY);
  cv::blur(workImage, workImage, cv::Size( 10, 10 ));
  cv::resize(workImage, workImage, cv::Size(), 0.25, 0.25);
  cv::threshold(workImage, workImage, 251, 255, cv::THRESH_BINARY);

  cv::resize(templateScaled, templateScaled, cv::Size(), 0.9, 0.9);
  cv::threshold(templateScaled, templateScaled, 251, 255, cv::THRESH_BINARY);

  // Create and configure generalized Hough transformation
  cv::Ptr<cv::GeneralizedHoughGuil> guil = cv::createGeneralizedHoughGuil();
  guil->setMinDist(100.0);
  guil->setLevels(1000);
  guil->setDp(2.0);
  guil->setMaxBufferSize(1000);

  guil->setMinAngle(0.0);
  guil->setMaxAngle(180.0);
  guil->setAngleStep(1.0);
  guil->setAngleThresh(1000);

  guil->setMinScale(0.1);
  guil->setMaxScale(1.1);
  guil->setScaleStep(0.05);
  guil->setScaleThresh(1000);

  guil->setPosThresh(100);
  guil->setTemplate(templateScaled);

  // Detect template in preprocessed image
  std::cout << "Start detecting" << std::endl;
  tm.start();
  guil->detect(workImage, detected);
  tm.stop();

  float angle = detected[0][3];

  // Debug output
  if(DEBUG) {
    for(int i = 0; i < detected.size(); i++) {
      std::cout << "Found : " << detected.size() << " objects" << std::endl;
      std::cout << "Detection time : " << tm.getTimeMilli() << " ms" << std::endl;

      cv::Point2f pos(detected[i][0], detected[i][1]);
      float scale = detected[i][2];

      // cvtColor(outputImage, outputImage, COLOR_GRAY2BGR);

      cv::RotatedRect rect;
      rect.center = pos;
      rect.size = cv::Size2f(templateImage.cols * scale, templateImage.rows * scale);
      rect.angle = angle;

      std::cout << "Position: " << detected[i][0] << " " << detected[i][1] << std::endl;
      std::cout << "Scale: " << scale << std::endl;
      std::cout << "Rotation: " << angle << std::endl;

      cv::Mat rotated = cv::getRotationMatrix2D(pos, angle, scale);

      // rotated.copyTo(outputImage);

      cv::Point2f pts[4];
      rect.points(pts);
      cv::Scalar color;

      if(i == 0) {
        color = cv::Scalar(0, 255, 0);
      } else {
        color = cv::Scalar(0, 0, 255);
      }

      cv::line(outputImage, pts[0], pts[1], cv::Scalar(0, 0, 255), 3);
      cv::line(outputImage, pts[1], pts[2], cv::Scalar(0, 0, 255), 3);
      cv::line(outputImage, pts[2], pts[3], cv::Scalar(0, 0, 255), 3);
      cv::line(outputImage, pts[3], pts[0], cv::Scalar(0, 0, 255), 3);
    }
    cv::imwrite("result.png", outputImage);
  }

  return angle;
}

float Image::matchTemplate(const std::string& pathToSearchImage, const std::string& pathToTemplateImage, const std::vector<std::vector<int>>& colorRange) {
  cv::Mat sourceImage = cv::imread(pathToSearchImage);
  cv::Mat templateImage = cv::imread(pathToTemplateImage, cv::IMREAD_GRAYSCALE);

  return matchTemplate(sourceImage, templateImage, colorRange);
}

//--------------------------------------------------------<
// Find shape in image and return position
std::vector<int> Image::findShape(const cv::Mat& image) {
  cv::Mat grayImage, blurImage, thresholdImage, kernel, openImage, closedImage, cannyImage;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Point> hull;
  std::vector<cv::Vec4i> hierarchy;
  std::vector<int> center;

  // Create threshold image
  cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
  cv::blur(grayImage, blurImage, cv::Size(5, 5));
  cv::threshold(blurImage, thresholdImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

  // Use morphology to clean the image
  kernel = cv::Mat::ones( 16, 16, CV_32F );
  cv::morphologyEx( thresholdImage, openImage, cv::MORPH_OPEN, kernel );
  cv::morphologyEx( openImage, closedImage, cv::MORPH_CLOSE, kernel );

  // Detect edges using canny edge detector
  cv::Canny( closedImage, cannyImage, 0, 1, 3 );

  // find contours
  cv::findContours(cannyImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

  // Find biggest contour
  sort(contours.begin(), contours.end(), compareContourAreas);
  std::vector<cv::Point> cnt = contours[contours.size()-1];

  // Get the convex hull and center
  hull = getHullFromContour(cnt);
  center = getCenterOfHull(hull);
  std::cout << center[0] << center[1] << std::endl;
  return center;
}

std::vector<int> Image::findShape(const std::string& pathToImage) {
  const cv::Mat image = cv::imread(pathToImage);
  return findShape(image);
}

//--------------------------------------------------------<
// Utility methods

// Replaces all areas within the provided color range with white
cv::Mat Image::removeColorRange(const cv::Mat& inputImage, const std::vector<std::vector<int>>& colorRange) {
  cv::Mat workingCopy = inputImage.clone();
  cv::Mat imageHSV;
  cv::Mat mask;

  // apply retrived color range on image
  cv::cvtColor(inputImage, imageHSV, cv::COLOR_BGR2HSV);
  cv::inRange(imageHSV, colorRange[0], colorRange[1], mask);
  workingCopy.setTo(cv::Scalar(255,255,255), mask);

  return workingCopy;
}

// Extracts areas within the provided color range and returns binarized mask containing these areas
cv::Mat Image::createColorRangeMask(const cv::Mat& image, const std::vector<std::vector<int>>& colorRange) {
  cv::Mat blurImage, imageHSV, mask;
  cv::cvtColor(image, imageHSV, cv::COLOR_BGR2HSV);
  cv::blur(imageHSV, blurImage, cv::Size(3, 3));
  cv::inRange(imageHSV, colorRange[0], colorRange[1], mask);
  mask =  cv::Scalar::all(255) - mask;
  return mask;
}

// Create and return a convex hull based on the biggest contour provided
std::vector<cv::Point> Image::getHullFromContour(const std::vector<cv::Point>& cnt) {
  std::vector<cv::Point> hull;

  cv::convexHull(cv::Mat(cnt), hull, false);
  return hull;
}

// Return the center of a provided convex hull
std::vector<int> Image::getCenterOfHull(const std::vector<cv::Point>& hull) {
  cv::Moments m;

  m = cv::moments(hull,true);
  std::vector<int> hullCenter{int(m.m10/m.m00), int(m.m01/m.m00)};

  return hullCenter;
}

// Crop and return image based on provided mask
cv::Mat Image::cropImageToMask(const cv::Mat& image, const cv::Mat& mask) {
  cv::Rect boundRect = findContainedRect(mask);
  return image(boundRect).clone();
}

// Extract bounding rect from picture
cv::Rect Image::findContainedRect(const cv::Mat& mask) {
  cv::Rect boundRect;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  std::vector<cv::Point> cnt;

  cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
  sort(contours.begin(), contours.end(), compareContourAreas);
  cnt = contours[contours.size()-1];

  boundRect = cv::boundingRect( cv::Mat(cnt) );

  return boundRect;
}

// Crop image to bouding
cv::Mat Image::cropImageToRect(const cv::Mat& image, const cv::Rect& boundRect) {
  return image(boundRect).clone();
}

// Sort contours by size
bool Image::compareContourAreas (std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
    double i = fabs(cv::contourArea(cv::Mat(contour1)));
    double j = fabs(cv::contourArea(cv::Mat(contour2)));
    return ( i < j );
}
