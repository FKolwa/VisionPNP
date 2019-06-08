#include "../include/image.h"

//--------------------------------------------------------<
// Match template in image and return rotation
float Image::matchTemplate(const cv::Mat& inputImage, const cv::Mat& templateImage, const std::vector<std::vector<int>>& colorRange, const std::string& configPath) {
  bool DEBUG = true;
  float scaleFactor = 0.25;
  float angle = 0.0;
  std::vector<cv::Vec4f> detected;
  cv::TickMeter tm;

  std::ifstream rawConfig(configPath);
  nlohmann::json config;
  rawConfig >> config;

  cv::Mat outputImage = inputImage.clone();
  cv::Mat templateScaled = templateImage.clone();
  cv::Mat searchImage;

  // In case the input image is already binarized
  if(inputImage.type() > 6) {
    searchImage = binaryFromRange(inputImage, colorRange);
  } else {
    searchImage = inputImage.clone();
  }

  // scale images to reduce amount of pixels to compare
  cv::resize(outputImage, outputImage, cv::Size(), scaleFactor, scaleFactor);

  // Use morphology to clean the image
  cv::Mat kernel = cv::Mat::ones( 8, 8, CV_32F );
  cv::morphologyEx( searchImage, searchImage, cv::MORPH_OPEN, kernel );
  cv::morphologyEx( searchImage, searchImage, cv::MORPH_CLOSE, kernel );
  cv::blur(searchImage, searchImage, cv::Size(6,6));
  // cv::blur(templateImage, templateImage, cv::Size(6,6));

  // Preprocess search image
  cv::resize(searchImage, searchImage, cv::Size(), scaleFactor, scaleFactor);
  if(DEBUG) {
    cv::Mat  templateCanny, searchCanny;
    cv::Canny( searchImage, searchCanny, 254, 255, 3 );
    cv::Canny( templateImage, templateCanny, 254, 255, 3 );
    cv::imwrite("./DEBUG_matchTemplate_01.png", searchImage);
    cv::imwrite("./DEBUG_matchTemplate_02.png", searchCanny);
    cv::imwrite("./DEBUG_matchTemplate_03.png", templateCanny);
  }

  // Preprocess template image
  // cv::cvtColor(templateScaled, templateScaled, cv::COLOR_BGR2GRAY);
  // cv::resize(templateScaled, templateScaled, cv::Size(), scaleFactor, scaleFactor);
  // cv::threshold(templateScaled, templateScaled, 251, 255, cv::THRESH_BINARY);

  // Create and configure generalized Hough transformation
  cv::Ptr<cv::GeneralizedHoughGuil> guil = cv::createGeneralizedHoughGuil();
  guil->setMinDist(config["MinDist"]);
  guil->setLevels(config["Levels"]);
  // guil->setDp(config["Dp"]);
  // guil->setCannyHighThresh(255);
  // guil->setCannyLowThresh(254);

  // guil->setMaxBufferSize(config["MaxBufferSize"]);
  // guil->setXi(config["Xi"]);
  guil->setMinAngle(config["MinAngle"]);
  guil->setMaxAngle(config["MaxAngle"]);
  guil->setAngleStep(config["AngleStep"]);
  guil->setAngleEpsilon(config["AngleEpsilon"]);
  // guil->setAngleThresh(config["AngleThresh"]);

  guil->setMinScale(config["MinScale"]);
  guil->setMaxScale(config["MaxScale"]);
  guil->setScaleStep(config["ScaleStep"]);
  // guil->setScaleThresh(config["ScaleThresh"]);

  // guil->setPosThresh(config["PosThresh"]);
  guil->setTemplate(templateImage, cv::Point(templateImage.cols/2, templateImage.rows/2));

  // Detect template in preprocessed image
  std::cout << "Start detecting" << std::endl;
  tm.start();
  // detect returns a vector of cv::Vec4f
  // The Vec4f contains [X-position, Y-position, scale, rotation]
  guil->detect(searchImage, detected);

  tm.stop();

  if(detected.size() > 0) {
  // candidate with highest vote at start of returned vector
  angle = detected[0][3];
    // Debug output
    if(DEBUG) {
      for(int i = 0; i < detected.size(); i++) {
        // Create rotated rect from position, scale and rotation
        cv::RotatedRect rect;
        rect.center = cv::Point2f(detected[i][0], detected[i][1]);
        rect.size = cv::Size2f(templateScaled.cols * detected[i][2], templateScaled.rows * detected[i][2]);
        rect.angle = detected[i][3];

        // Create edge points from rotated rectangle
        cv::Point2f pts[4];
        rect.points(pts);

        // Draw bouding box
        cv::Scalar color = (i == 0 ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 0));
        cv::line(outputImage, pts[0], pts[1], color, 3);
        cv::line(outputImage, pts[1], pts[2], color, 3);
        cv::line(outputImage, pts[2], pts[3], color, 3);
        cv::line(outputImage, pts[3], pts[0], color, 3);

      }
      std::cout << "Found : " << detected.size() << " objects" << std::endl;
      std::cout << "Detection time : " << tm.getTimeMilli() << " ms" << std::endl;
      std::cout << "Position: " << detected[0][0] << " " << detected[0][1] << std::endl;
      std::cout << "Scale: " << detected[0][2] << std::endl;
      std::cout << "Rotation: " << detected[0][3] << std::endl;
      cv::imwrite("DEBUG_matchTemplate_04.png", outputImage);
    }
  } else {
    std::cout << "Could not detect template in search image" << std::endl;
  }

  return angle;
}

float Image::matchTemplate(const std::string& pathToSearchImage, const std::string& pathToTemplateImage, const std::vector<std::vector<int>>& colorRange, const std::string& configPath) {
  cv::Mat sourceImage = cv::imread(pathToSearchImage);
  cv::Mat templateImage = cv::imread(pathToTemplateImage, cv::IMREAD_GRAYSCALE);
  return matchTemplate(sourceImage, templateImage, colorRange, configPath);
}

//--------------------------------------------------------<
// Find shape in image and return position
std::vector<int> Image::findShape(const cv::Mat& image) {
  bool DEBUG = false;
  cv::Mat grayImage, blurImage, thresholdImage, kernel, openImage, closedImage, cannyImage;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Point> hull;
  std::vector<cv::Vec4i> hierarchy;
  std::vector<int> center;

  // Create threshold image
  if(image.type() > 6) { // More than one channel
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
  } else {
    grayImage = image.clone();
  }

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
  std::vector<std::vector<cv::Point>> dummyVector;
  dummyVector.push_back(cnt);

  // Get the convex hull and center
  hull = getHullFromContour(cnt);
  center = getCenterOfHull(cnt);

  std::vector<std::vector<cv::Point>> dummyVectorHull;
  dummyVectorHull.push_back(hull);

  if(DEBUG) {
    // Draw contours for debugging
    drawContours(contours, hierarchy, image, "./DEBUG_findShape_contours_01.png");
    drawContours(dummyVector, hierarchy, image, "./DEBUG_findShape_contours_02.png");
    drawContours(dummyVectorHull, hierarchy, image, "./DEBUG_findShape_contours_03.png");
  }

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

// Binarizes image based on color range
cv::Mat Image::binaryFromRange(const cv::Mat& inputImage, const std::vector<std::vector<int>>& colorRange) {
  // cv::Mat workingCopy(inputImage.rows, inputImage.cols, CV_8UC1, 0);
  // cv::Mat workingCopy = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC1);
  cv::Mat black(inputImage.rows, inputImage.cols, CV_8UC1, 0);
  // cv::Mat white(inputImage.rows, inputImage.cols, CV_8UC1, 255);
  // cv::Mat workingCopy = inputImage.clone();
  // workingCopy.create( cv::Size(inputImage.cols, inputImage.rows), CV_8UC3);
  cv::Mat imageHSV;
  cv::Mat mask;

  // apply retrived color range on image
  cv::cvtColor(inputImage, imageHSV, cv::COLOR_BGR2HSV);
  cv::inRange(imageHSV, colorRange[0], colorRange[1], mask);
  //workingCopy.setTo(255, mask);
  // inputImage.copyTo(workingCopy, mask);
  black.setTo(255, mask);
  //mask.copyTo(white, black);
  cv::imwrite("./maskimage.png", black);
  // workingCopy.setTo(0, mask);

  return black;
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

// Draw contours on image
void Image::drawContours(std::vector<std::vector<cv::Point>> contours, std::vector<cv::Vec4i> hierarchy, cv::Mat inputImage, std::string outputPath) {
  cv::Mat outputImage = inputImage.clone();
  cv::Scalar color = cv::Scalar( 0, 0, 255 );
  for( int i = 0; i< contours.size(); i++ ) {
    cv::drawContours( outputImage, contours, i, color, 2, 8, std::vector<cv::Vec4i>(), 0, cv::Point() );
  }
  cv::imwrite(outputPath, outputImage);
}

std::string Image::getMatType(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}
