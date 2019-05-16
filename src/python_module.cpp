#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/color.h"
#include "../include/image.h"

using namespace std;
namespace py = pybind11;

typedef unsigned char BYTE;


string type2str(int type) {
  string r;

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


// Mat (int ndims, const int *sizes, int type, void *data, const size_t *steps=0)
// Mat (long int&, cv::Size*, int, void*&, __gnu_cxx::__alloc_traits<std::allocator<long int>, long int>::value_type*)’



// static Mat numpyToMat(py::array imgArray) {
//   auto info = imgArray.request();
//   decltype(CV_32F) dtype;
//   if(info.format == py::format_descriptor<float>::value) dtype = CV_32F;
//   else if (info.format == py::format_descriptor<double>::value) dtype = CV_64F;


//   auto ndims = info.ndim;
//   Size shape (info.shape[0], info.shape[1]);
//   auto& strides = info.strides;
//   Mat image(int(&ndims), &shape, CV_32F, info.ptr, &strides[0]);
//   return image;
// }


    // bool load(handle src, bool) {
    //     array b(src, true);
    //     if(!b.check()) return false;
    //     auto info = b.request();

    //     decltype(CV_32F) dtype;
    //     if(info.format == format_descriptor<float>::value) dtype = CV_32F;
    //     else if (info.format == format_descriptor<double>::value) dtype = CV_64F;
    //     else return false;

    //     auto ndims = info.ndim;
    //     auto shape = std::vector<int>(info.shape.begin(), info.shape.end());
    //     auto& strides = info.strides;
    //     value = cv::Mat(ndims,
    //             &shape[0],
    //             dtype,
    //             info.ptr,
    //             &strides[0]);
    //     return true;

    // }

    // static handle cast(const cv::Mat &m, return_value_policy, handle defval) {
    //     auto format = format_descriptor<float>::value;
    //     auto type = m.type();
    //     switch(type) {
    //         case CV_32F: format = format_descriptor<float>::value; break;
    //         case CV_64F: format = format_descriptor<double>::value; break;
    //         default: return defval;
    //     }

    //     std::vector<size_t> IHateBjarneStroustrup;
    //     std::copy(m.size.p, m.size.p + m.dims, std::back_inserter(IHateBjarneStroustrup));
    //     auto strides = std::vector<size_t>(m.step.p, m.step.p + m.dims);
    //     strides.push_back(1);
    //     return array(buffer_info(
    //         m.data,
    //         m.elemSize1(),
    //         format,
    //         m.dims,
    //         IHateBjarneStroustrup,
    //         strides
    //         )).release();
    // }

//----------------------------------------------------------<
// numpy array to Mat converter
static Mat numpyToMat(py::array input) {
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

  Mat img(Size(width, height), type, ptr, Mat::AUTO_STEP);
  return img;
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
  return Image::cropImageToMask(matImage, matMask);
}

static Mat pyCreateColorRangeMask(const py::array& image, const vector <vector<int>>& colorRange) {
  Mat matImage = numpyToMat(image);
  return Image::createColorRangeMask(matImage, colorRange);
}

static vector<int> pyFindShape(const py::array& image) {
  Mat matImage = numpyToMat(image);
  return Image::findShape(matImage);
}

static vector<int> pyFindShape(const string& imagePath) {
  return Image::findShape(imagePath);
}

//----------------------------------------------------------<
// Python module definition
PYBIND11_MODULE(VisionPNP, m){
  m.doc() = "Tool for detecting objects in images.";

  m.def("numpyToMat", &numpyToMat);

  m.def("getHSVColorRange", &Color::getHSVColorRange,
    "Returns lower and upper color ranges from provided background picture.",
    py::arg("imagePath"));

  // m.def("matchTemplate", &Image::matchTemplate,
  //   "Detects and retrieves most likely candidate of provided template in search image.",
  //   py::arg("pathToSearchImage"),
  //   py::arg("pathToTemplateImage"),
  //   py::arg("colorRange"));

  m.def("findShape", py::overload_cast<const py::array&>(&pyFindShape), "Detects arbitrary shape in provided search image.", py::arg("image"));

  m.def("findShape", py::overload_cast<const string&>(&pyFindShape), "Detects arbitrary shape in provided search image.", py::arg("imagePath"));

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
