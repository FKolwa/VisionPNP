#include <vector>
#include <iostream>
#include <string>

#ifndef OPENCV_H
#define OPENCV_H
#include "opencv2/opencv.hpp"
#endif

class Image {
  public:
    static float matchTemplate(const std::string& pathToSearchImage, const std::string& pathToTemplateImage, const std::vector<std::vector<int>>& colorRange);
    static float matchTemplate(const cv::Mat& searchImage, const cv::Mat& templateImage, const std::vector<std::vector<int>>& colorRange);
    static std::vector<int> findShape(const std::string& pathToImage);
    static std::vector<int> findShape(const cv::Mat& searchImage);
    static cv::Mat removeColorRange(const cv::Mat& inputImage, const std::vector<std::vector<int>>& colorRange);
    static cv::Rect findContainedRect(const cv::Mat& mask);
    static cv::Mat cropImageToRect(const cv::Mat& image, const cv::Rect& boudingRect);
    static cv::Mat cropImageToMask(const cv::Mat& image, const cv::Mat& mask);
    static cv::Mat createColorRangeMask(const cv::Mat& image, const std::vector <std::vector<int>>& colorRange);
  private:
    static std::vector<cv::Point> getHullFromContour(const std::vector<cv::Point>& contours);
    static std::vector<int> getCenterOfHull(const std::vector<cv::Point>& hull);
    static bool compareContourAreas (std::vector<cv::Point> contour1, std::vector<cv::Point> contour2);
};
