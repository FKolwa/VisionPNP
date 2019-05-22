#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/color.h"
#include "../include/image.h"

namespace py = pybind11;

std::string type2str(int type) {
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

//----------------------------------------------------------<
// numpy array to Mat converter
static cv::Mat numpyToMat(py::array input) {
  py::buffer_info info = input.request();
  unsigned int width = info.shape[0];
  unsigned int height = info.shape[1];
  auto ptr = static_cast<unsigned char *>(info.ptr);
  int type = 0;
  switch(info.shape[2]) {
    case 1: type = CV_8UC1;
      break;
    case 2: type = CV_8UC2;
      break;
    case 3: type = CV_8UC3;
      break;
    default: type = CV_8UC1;
  }

  cv::Mat img(cv::Size(width, height), type, ptr, cv::Mat::AUTO_STEP);
  return img;
}

//----------------------------------------------------------<
// Wrappers for methods that expect Mat objects as arguments
static cv::Mat pyRemoveColorRange(const py::array& inputImage, const std::vector<std::vector<int>>& colorRange) {
  cv::Mat image = numpyToMat(inputImage);
  return Image::removeColorRange(image, colorRange);
}

static cv::Mat pyCropImageToMask(const py::array& image, const py::array& mask) {
  cv::Mat matImage = numpyToMat(image);
  cv::Mat matMask = numpyToMat(mask);
  return Image::cropImageToMask(matImage, matMask);
}

static cv::Mat pyCreateColorRangeMask(const py::array& image, const std::vector<std::vector<int>>& colorRange) {
  cv::Mat matImage = numpyToMat(image);
  return Image::createColorRangeMask(matImage, colorRange);
}

static std::vector<int> pyFindShape(const py::array& image) {
  cv::Mat matImage = numpyToMat(image);
  return Image::findShape(matImage);
}

static std::vector<int> pyFindShape(const std::string& imagePath) {
  return Image::findShape(imagePath);
}

static float pyMatchTemplate(const py::array& inputImage, const py::array& templateImage, const std::vector<std::vector<int>>& colorRange) {
  cv::Mat imageMat = numpyToMat(inputImage);
  cv::Mat tempMat = numpyToMat(templateImage);
  return Image::matchTemplate(imageMat, tempMat, colorRange);
}

static float pyMatchTemplate(const std::string& imagePath, const std::string& templatePath, const std::vector<std::vector<int>>& colorRange) {
  return Image::matchTemplate(imagePath, templatePath, colorRange);
}

static std::vector<int> pyFindContainedRect(const py::array& inputImage) {
  cv::Mat imageMat = numpyToMat(inputImage);
  cv::Rect bRect = Image::findContainedRect(imageMat);
  return std::vector<int> {bRect.height, bRect.width, bRect.x, bRect.y};
}

static cv::Mat pyCropImageToRect(const py::array& inputImage, const std::vector<int> boudingRect) {
  cv::Mat imageMat = numpyToMat(inputImage);
  cv::Rect bRect;
  bRect.height = boudingRect[0];
  bRect.width = boudingRect[1];
  bRect.x = boudingRect[2];
  bRect.y = boudingRect[3];
  return Image::cropImageToRect(imageMat, bRect);
}
//----------------------------------------------------------<
// Python module definition

void initPythonBindings(py::module& m) {
  m.def("numpyToMat", &numpyToMat);

  m.def("getHSVColorRange", &Color::getHSVColorRange,
    "Returns lower and upper color ranges from provided background picture.",
    py::arg("imagePath"));

  m.def("matchTemplate",
    py::overload_cast<const py::array&, const py::array&, const std::vector<std::vector<int>>&>(&pyMatchTemplate),
    "Detects and retrieves most likely candidate of provided template in search image.",
    py::arg("inputImage"),
    py::arg("templateImage"),
    py::arg("colorRange"));

  m.def("matchTemplate",
    py::overload_cast<const std::string&, const std::string&, const std::vector<std::vector<int>>&>(&pyMatchTemplate),
    "Detects and retrieves most likely candidate of provided template in search image.",
    py::arg("imagePath"),
    py::arg("templatePath"),
    py::arg("colorRange"));

  m.def("findShape",
    py::overload_cast<const py::array&>(&pyFindShape),
    "Detects arbitrary shape in provided search image.",
    py::arg("inputImage"));

  m.def("findShape",
    py::overload_cast<const std::string&>(&pyFindShape),
    "Detects arbitrary shape in provided search image.",
    py::arg("imagePath"));

  m.def("removeColorRange", &pyRemoveColorRange,
    "Removes provided HSV color range from picture",
    py::arg("inputImage"),
    py::arg("colorRange"));

  m.def("cropImageToMask", &pyCropImageToMask,
    "Crops image to innersize of biggest contour in provided binary mask.",
    py::arg("inputImage"),
    py::arg("mask"));

  m.def("createColorRangeMask", &pyCreateColorRangeMask,
    "Creates binary mask only containing elements contained in provided color range.",
    py::arg("inputImage"),
    py::arg("colorRange"));

  m.def("findContainedRect", &pyFindContainedRect,
    "Extracts bouding rect from shape contained in mask.",
    py::arg("mask"));

  m.def("cropImageToRect", &pyCropImageToRect,
    "Crops image to bounding rect.",
    py::arg("inputImage"),
    py::arg("boundingRect"));

  py::class_<cv::Mat>(m, "pyMat", py::buffer_protocol())
    .def_buffer([](cv::Mat& im) -> py::buffer_info {
      return py::buffer_info(
        im.data,
        sizeof(unsigned char),
        py::format_descriptor<unsigned char>::format(),
        3,
        { im.rows, im.cols, im.channels() },
        {
          sizeof(unsigned char) * im.channels() * im.cols,
          sizeof(unsigned char) * im.channels(),
          sizeof(unsigned char)
        }
      );
    });
}

// pybind11 legacy fallback
#ifndef PYBIND11_MODULE
PYBIND11_PLUGIN(VisionPNP) {
  py::module m("VisionPNP", "python plugin for cv pick and place automation");
  initPythonBindings(m);
  return m.ptr;
}
#else
PYBIND11_MODULE(VisionPNP, m) {
  m.doc() = "python plugin for cv pick and place automation";
  initPythonBindings(m);
}
#endif
