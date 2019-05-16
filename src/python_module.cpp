#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/color.h"
#include "../include/image.h"

using namespace std;
namespace py = pybind11;

typedef unsigned char BYTE;

//----------------------------------------------------------<
// numpy array to Mat converter
static Mat numpyToMat(py::array xs) {
  py::buffer_info info = xs.request();
  unsigned int width = info.shape[0];
  unsigned int height = info.shape[1];
  auto ptr = static_cast<unsigned char *>(info.ptr);

  Mat img(Size(width, height), CV_8UC3, ptr, Mat::AUTO_STEP);
  Mat image = img.clone(); // workaround for correct image convertion
  return image;
}

//----------------------------------------------------------<
// Wrappers for methods that expect Mat objects as arguments
static Mat pyRemoveColorRange(const py::array& inputImage, const vector <vector<int>>& colorRange) {
  Mat image = numpyToMat(inputImage);
  return Image::removeColorRange(image, colorRange);
}

static Mat pyCropImageToMask(const py::array& image, const py::array& mask) {
  Mat matImage = numpyToMat(image);
  Mat matMask = numpyToMat(mask);
  cvtColor(matMask, matMask, COLOR_BGR2GRAY);
  return Image::cropImageToMask(matImage, matMask);
}

static Mat pyCreateColorRangeMask(const py::array& image, const vector <vector<int>>& colorRange) {
  Mat matImage = numpyToMat(image);
  return Image::createColorRangeMask(matImage, colorRange);
}

//----------------------------------------------------------<
// Python module definition
PYBIND11_MODULE(VisionPNP, m){
  m.doc() = "Tool for detecting objects in images.";

  m.def("numpyToMat", &numpyToMat);

  m.def("getHSVColorRange", &Color::getHSVColorRange,
    "Returns lower and upper color ranges from provided background picture.",
    py::arg("imagePath"));

  m.def("matchTemplate", &Image::matchTemplate,
    "Detects and retrieves most likely candidate of provided template in search image.",
    py::arg("pathToSearchImage"),
    py::arg("pathToTemplateImage"),
    py::arg("colorRange"));

  m.def("findShape", &Image::findShape,
    "Detects arbitrary shape in provided search image.",
    py::arg("pathToImage"));

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
