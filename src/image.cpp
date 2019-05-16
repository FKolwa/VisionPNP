#include "../include/image.h"

//--------------------------------------------------------<
// Match template in image and return rotation
float Image::matchTemplate(const Mat& inputImage,const Mat& templateImage, const vector <vector<int>>& colorRange) {
  bool DEBUG = false;
  vector<Vec4f> detected;
  TickMeter tm;
  Mat workImage;
  Mat outputImage = inputImage.clone();

  workImage = removeColorRange(inputImage, colorRange);

  // binarize image
  cvtColor(workImage, workImage, COLOR_BGR2GRAY);
  threshold(workImage, workImage, 0, 255, THRESH_BINARY | THRESH_OTSU);


  // Preprocess raw image data
  resize(workImage, workImage, Size(), 0.25, 0.25);
  blur(workImage, workImage, Size( 10, 10 ));
  threshold(workImage, workImage, 0, 255, THRESH_BINARY | THRESH_OTSU);

  // Create and configure generalized Hough transformation
  Ptr<GeneralizedHoughGuil> guil = createGeneralizedHoughGuil();
  guil->setMinDist(100.0);
  guil->setLevels(360);
  guil->setDp(2.0);
  guil->setMaxBufferSize(1000);

  guil->setMinAngle(0.0);
  guil->setMaxAngle(180.0);
  guil->setAngleStep(1.0);
  guil->setAngleThresh(1000);

  guil->setMinScale(0.1);
  guil->setMaxScale(10);
  guil->setScaleStep(0.05);
  guil->setScaleThresh(1000);

  guil->setPosThresh(100);
  guil->setTemplate(templateImage);

  // Detect template in preprocessed image
  tm.start();
  guil->detect(workImage, detected);
  tm.stop();

  float angle = detected[0][3];

  // Debug output
  if(DEBUG) {
    cout << "Found : " << detected.size() << " objects" << endl;
    cout << "Detection time : " << tm.getTimeMilli() << " ms" << endl;


    Point2f pos(detected[0][0], detected[0][1]);
    float scale = detected[0][2];

    cvtColor(outputImage, outputImage, COLOR_GRAY2BGR);

    RotatedRect rect;
    rect.center = pos;
    rect.size = Size2f(templateImage.cols * scale, templateImage.rows * scale);
    rect.angle = angle;

    cout << "Position: " << detected[0][0] << " " << detected[0][1] << endl;
    cout << "Scale: " << scale << endl;
    cout << "Rotation: " << angle << endl;

    Point2f pts[4];
    rect.points(pts);

    line(outputImage, pts[0], pts[1], Scalar(0, 0, 255), 3);
    line(outputImage, pts[1], pts[2], Scalar(0, 0, 255), 3);
    line(outputImage, pts[2], pts[3], Scalar(0, 0, 255), 3);
    line(outputImage, pts[3], pts[0], Scalar(0, 0, 255), 3);

    imwrite("result.png", outputImage);
  }

  return angle;
}

float Image::matchTemplate(const string& pathToSearchImage, const string& pathToTemplateImage, const vector <vector<int>>& colorRange) {
  Mat sourceImage = imread(pathToSearchImage);
  Mat templateImage = imread(pathToTemplateImage, IMREAD_GRAYSCALE);

  return matchTemplate(sourceImage, templateImage, colorRange);
}

//--------------------------------------------------------<
// Find shape in image and return position
vector<int> Image::findShape(const Mat& image) {
  Mat grayImage, blurImage, thresholdImage, kernel, openImage, closedImage, cannyImage;
  vector<vector<Point>> contours;
  vector <Point> hull;
  vector<Vec4i> hierarchy;
  vector<int> center;

  // Create threshold image
  cvtColor(image, grayImage, COLOR_BGR2GRAY);
  blur(grayImage, blurImage, Size(5, 5));
  threshold(blurImage, thresholdImage, 0, 255, THRESH_BINARY | THRESH_OTSU);

  // Use morphology to clean the image
  kernel = Mat::ones( 16, 16, CV_32F );
  morphologyEx( thresholdImage, openImage, MORPH_OPEN, kernel );
  morphologyEx( openImage, closedImage, MORPH_CLOSE, kernel );

  // Detect edges using canny edge detector
  Canny( closedImage, cannyImage, 0, 1, 3 );

  // find contours
  findContours(cannyImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

  // Find biggest contour
  sort(contours.begin(), contours.end(), compareContourAreas);
  vector <Point> cnt = contours[contours.size()-1];

  // Get the convex hull and center
  hull = getHullFromContour(cnt);
  center = getCenterOfHull(hull);

  return center;
}

vector<int> Image::findShape(const string& pathToImage) {
  const Mat image = imread(pathToImage);
  return findShape(image);
}

//--------------------------------------------------------<
// Utility methods

// Replaces all areas within the provided color range with white
Mat Image::removeColorRange(const Mat& inputImage, const vector <vector<int>>& colorRange) {
  Mat workingCopy = inputImage.clone();
  Mat imageHSV;
  Mat mask;

  // apply retrived color range on image
  cvtColor(inputImage, imageHSV, COLOR_BGR2HSV);
  inRange(imageHSV, colorRange[0], colorRange[1], mask);
  workingCopy.setTo(Scalar(255,255,255), mask);

  return workingCopy;
}

// Extracts areas within the provided color range and returns binarized mask containing these areas
Mat Image::createColorRangeMask(const Mat& image, const vector <vector<int>>& colorRange) {
  Mat blurImage, imageHSV, mask;
  cvtColor(image, imageHSV, COLOR_BGR2HSV);
  blur(imageHSV, blurImage, Size(3, 3));
  inRange(imageHSV, colorRange[0], colorRange[1], mask);
  mask =  Scalar::all(255) - mask;
  return mask;
}

// Create and return a convex hull based on the biggest contour provided
vector<Point> Image::getHullFromContour(const vector<Point>& cnt) {
  vector<Point> hull;

  convexHull(Mat(cnt), hull, false);
  return hull;
}

// Return the center of a provided convex hull
vector<int> Image::getCenterOfHull(const vector<Point>& hull) {
  Moments m;

  m = moments(hull,true);
  vector<int> hullCenter{int(m.m10/m.m00), int(m.m01/m.m00)};

  return hullCenter;
}

// Crop and return image based on provided mask
Mat Image::cropImageToMask(const Mat& image, const Mat& mask) {
  Rect bRect;
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  vector<Point> cnt;

  findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
  sort(contours.begin(), contours.end(), compareContourAreas);
  cnt = contours[contours.size()-1];

  vector<Point> contours_poly;
  Rect boundRect;

  boundRect = boundingRect( Mat(cnt) );

  /// Draw bonding rect
  // Mat drawing = image.clone();
  // Scalar color = Scalar(0, 0, 255);
  // rectangle( drawing, boundRect.tl(), boundRect.br(), color, 2, 8, 0 );
  // imwrite("output_cropped.png", image(boundRect));

  return image(boundRect).clone();
}

// Sort contours by size
bool Image::compareContourAreas (vector<Point> contour1, vector<Point> contour2) {
    double i = fabs(contourArea(Mat(contour1)));
    double j = fabs(contourArea(Mat(contour2)));
    return ( i < j );
}
