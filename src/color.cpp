#include "../include/color.h"

vector <vector<int>> Color::getHSVColorRange(const string& imagePath) {
  vector <vector<int>> threshold;
  Mat inputImage = imread(imagePath);
  Mat workingCopy = inputImage.clone();
  Mat colorMatrix;

  // Convert to HSV spectrum and read smoothed colors from image
  cvtColor(workingCopy, workingCopy, COLOR_BGR2HSV);
  blur(workingCopy, workingCopy, Size( 8, 8 ));
  colorMatrix = readColors(workingCopy);

  // Calculates mean and standard deviation of all color elements.
  Scalar mean,dev;
  meanStdDev(colorMatrix,mean,dev);
  threshold.push_back(vector<int> {int(mean[0]-dev[0]*4), int(mean[1]-dev[1]*4), int(mean[2]-dev[2]*4)});
  threshold.push_back(vector<int> {int(mean[0]+dev[0]*4), int(mean[1]+dev[1]*4), int(mean[2]+dev[2]*4)});

  return threshold;
}

Mat Color::readColors(const Mat& image) {
  Mat colors;
  int offset = 50;
  int cX = (image.cols/2)-offset;
  int cY = (image.rows/2)-offset;
  double min = 180;
  double max = 0;

  for (int x=0; x<image.cols; x++) {
    for (int y=0; y<image.rows; y++) {
      if ((x < cX || x > cX + offset*2) && (y < cY || y > cY + offset*2 )) {
        Vec3b p = image.at<Vec3b>(y, x);
        colors.push_back(p);
      }
    }
  }
  return colors;
}
