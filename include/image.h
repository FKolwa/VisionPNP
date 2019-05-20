#include <vector>
#include <iostream>
#include <string>

#ifndef OPENCV_H
#define OPENCV_H
#include "opencv2/opencv.hpp"
#endif

using namespace std;
using namespace cv;

class Image {
  public:
    static float matchTemplate(const string& pathToSearchImage, const string& pathToTemplateImage, const vector <vector<int>>& colorRange);
    static float matchTemplate(const Mat& searchImage, const Mat& templateImage, const vector <vector<int>>& colorRange);
    static vector<int> findShape(const string& pathToImage);
    static vector<int> findShape(const Mat& searchImage);
    static Mat removeColorRange(const Mat& inputImage, const vector <vector<int>>& colorRange);
    static Rect findContainedRect(const Mat& mask);
    static Mat cropImageToRect(const Mat& image, const Rect& boudingRect);
    static Mat cropImageToMask(const Mat& image, const Mat& mask);
    static Mat createColorRangeMask(const Mat& image, const vector <vector<int>>& colorRange);
  private:
    static vector<Point> getHullFromContour(const vector<Point>& contours);
    static vector<int> getCenterOfHull(const vector<Point>& hull);
    static bool compareContourAreas (vector<Point> contour1, vector<Point> contour2);
};
