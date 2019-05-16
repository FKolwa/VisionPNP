#include <iostream>
#include <tuple>

#ifndef OPENCV_H
#define OPENCV_H
#include "opencv2/opencv.hpp"
#endif

using namespace cv;
using namespace std;

class Color {
  public:
    static vector <vector<int>> getHSVColorRange(const string& imagePath);
  private:
    static Mat readColors(const Mat& image) ;
};
