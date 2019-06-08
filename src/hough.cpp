#include "../include/hough.h"

// Initialize with defaults
std::vector<std::vector<cv::Vec2i>> Hough::Rtable;
std::vector<Rpoint> Hough::pts;
cv::Mat Hough::accum;
int Hough::wmin = 0;
int Hough::wmax = 0;
float Hough::phimin = -PI/2;
float Hough::phimax = PI/2;
int Hough::rangeXY = 1;
int Hough::rangeS = 1;
int Hough::intervals = 128;
int Hough::wtemplate = 0;

// save file with canny edge of the original image
void Hough::createRtable(const cv::Mat& templateImage){
  cv::Mat cannyImage = templateImage.clone();
  cv::imwrite("./dummyloadedshit.png", cannyImage);
  std::cout << "gray" << std::endl;
  if(cannyImage.type() > 6) {
    cv::cvtColor(cannyImage, cannyImage, cv::COLOR_BGR2GRAY);
  }
  std::cout << "blut" << std::endl;
  cv::blur(cannyImage, cannyImage, cv::Size(3,3) );
  std::cout << "canny" << std::endl;
  cv::Canny(cannyImage, cannyImage, 50, 100);


  std::cout << "readPoints" << std::endl;
  readPoints(templateImage, cannyImage);
  std::cout << "readRtable" << std::endl;
  readRtable();
}

// fill accumulator matrix
void Hough::accumulate(const cv::Mat& searchImage){
  // transform image to grayscale:
  cv::Mat src_gray;
  src_gray.create( cv::Size(searchImage.cols, searchImage.rows), CV_8UC1);

  if(searchImage.type() > 6) {
    cv::cvtColor(searchImage, src_gray, cv::COLOR_BGR2GRAY);
  }

  // reduce noise with a kernel 3x3 and get cannyedge image:
  cv::Mat detected_edges;
  cv::blur( src_gray, detected_edges, cv::Size(3,3) );
  cv::Canny( detected_edges, detected_edges, 0, 1, 3 );

  // get Scharr matrices from image to obtain contour gradients
  cv::Mat dx;
  dx.create( cv::Size(searchImage.cols, searchImage.rows), CV_16SC1);
  cv::Sobel(src_gray, dx, CV_16S, 1, 0, cv::FILTER_SCHARR);
  cv::Mat dy;
  dy.create( cv::Size(searchImage.cols, searchImage.rows), CV_16SC1);
  cv::Sobel(src_gray, dy, CV_16S, 0, 1, cv::FILTER_SCHARR);

  // load all points from image all image contours on vector pts2
  int nl= detected_edges.rows;
  int nc= detected_edges.cols;
  float deltaphi = PI/intervals;
  float inv_deltaphi = (float)intervals/PI;
  float inv_rangeXY = (float)1/rangeXY;
  float PI_half = PI*0.5f;
  std::vector<Rpoint2> pts2;
  for (int j=0; j<nl; ++j) {
    uchar* data= (uchar*)(detected_edges.data + detected_edges.step.p[0]*j);
    for (int i=0; i<nc; ++i) {
      if ( data[i]==255  ) // consider only white points (contour)
      {
        short vx = dx.at<short>(j,i);
        short vy = dy.at<short>(j,i);
        Rpoint2 rpt;
        rpt.x = i*inv_rangeXY;
        rpt.y = j*inv_rangeXY;
        float a = atan2((float)vy, (float)vx);              //	gradient angle in radians
        float phi = ((a > 0) ? a-PI_half : a+PI_half);      // contour angle with respect to x axis
        int angleindex = (int)((phi+PI*0.5f)*inv_deltaphi); // index associated with angle (0 index = -90 degrees)
        if (angleindex == intervals) angleindex=intervals-1;// -90�angle and +90� has same effect
        rpt.phiindex = angleindex;
        pts2.push_back( rpt );
      }
    }
  }

  // OpenCv 4-dimensional matrix definition and in general a useful way for defining multidimensional arrays and vectors in c++
  // create accumulator matrix
  int X = ceil((float)nc/rangeXY);
  int Y = ceil((float)nl/rangeXY);
  int S = ceil((float)(wmax-wmin)/rangeS+1.0f);
  int R = ceil(phimax/deltaphi)-floor(phimin/deltaphi);
  if (phimax==PI && phimin==-PI) R--;
  int r0 = -floor(phimin/deltaphi);
  int matSizep_S[] = {X, Y, S, R};
  accum.create(4, matSizep_S, CV_16S);
  accum = cv::Scalar::all(0);

  // icrease accum cells with hits corresponding with slope in Rtable vector rotatated and scaled
  float inv_wtemplate_rangeXY = (float)1/(wtemplate*rangeXY);

  // rotate RTable from minimum to maximum angle
  for (int r=0; r<R; ++r) {  // rotation
    int reff = r-r0;
    std::vector<std::vector<cv::Vec2f>> Rtablerotated(intervals);
    // cos and sin are computed in the outer loop to reach computational efficiency
    float cs = cos(reff*deltaphi);
    float sn = sin(reff*deltaphi);
    for (std::vector<std::vector<cv::Vec2i>>::size_type ii = 0; ii < Rtable.size(); ++ii){
      for (std::vector<cv::Vec2i>::size_type jj= 0; jj < Rtable[ii].size(); ++jj){
        int iimod = (ii+reff) % intervals;
        Rtablerotated[iimod].push_back(cv::Vec2f(cs*Rtable[ii][jj][0] - sn*Rtable[ii][jj][1], sn*Rtable[ii][jj][0] + cs*Rtable[ii][jj][1]));
      }
    }
    // scale the rotated RTable from minimum to maximum scale
    for (int s=0; s<S; ++s) {  // scale
      std::vector<std::vector<cv::Vec2f>> Rtablescaled(intervals);
      int w = wmin + s*rangeS;
      float wratio = (float)w*inv_wtemplate_rangeXY;
      for (std::vector<std::vector<cv::Vec2f>>::size_type ii = 0; ii < Rtablerotated.size(); ++ii){
        for (std::vector<cv::Vec2f>::size_type jj= 0; jj < Rtablerotated[ii].size(); ++jj){
          Rtablescaled[ii].push_back(cv::Vec2f(wratio*Rtablerotated[ii][jj][0], wratio*Rtablerotated[ii][jj][1]));
        }
      }
      // iterate through each point of edges and hit corresponding cells from rotated and scaled Rtable
      for (std::vector<Rpoint2>::size_type t = 0; t < pts2.size(); ++t){ // XY plane
        int angleindex = pts2[t].phiindex;
        for (std::vector<cv::Vec2f>::size_type index = 0; index < Rtablescaled[angleindex].size(); ++index){
          float deltax = Rtablescaled[angleindex][index][0];
          float deltay = Rtablescaled[angleindex][index][1];
          int xcell = (int)(pts2[t].x + deltax);
          int ycell = (int)(pts2[t].y + deltay);
          if ( (xcell<X)&&(ycell<Y)&&(xcell>-1)&&(ycell>-1) ){
            // increment the correspconding elment in the H table by 1
            (*ptrat4D(accum, xcell, ycell, s, r))++;

          }
        }
      }
    }
  }
}

// show the best candidate detected on image
float Hough::bestCandidate(const cv::Mat& searchImage){
  cv::Mat showImage;

  if(searchImage.type() < 6) {
    cv::cvtColor(searchImage, showImage, cv::COLOR_GRAY2BGR);
  } else {
    showImage = searchImage.clone();
  }

  double minval;
  double maxval;
  int id_min[4] = { 0, 0, 0, 0};
  int id_max[4] = { 0, 0, 0, 0};
  cv::minMaxIdx(accum, &minval, &maxval, id_min, id_max);

  int nl= showImage.rows;
  int nc= showImage.cols;

  cv::Vec2i referenceP = cv::Vec2i(id_max[0]*rangeXY+(rangeXY+1)/2, id_max[1]*rangeXY+(rangeXY+1)/2);

  // std::cout << "Position: " << id_max[0]*rangeXY+(rangeXY+1)/2 << " - "<< id_max[1]*rangeXY+(rangeXY+1)/2 << std::endl;

  // rotate and scale points all at once. Then impress them on image
  // std::vector<std::vector<Vec2i>> Rtablerotatedscaled(intervals);
  float deltaphi = PI/intervals;
  int r0 = -floor(phimin/deltaphi);
  int reff = id_max[3]-r0;
  float angle = reff*deltaphi;
  float cs = cos(angle);
  float sn = sin(angle);
  int w = wmin + id_max[2]*rangeS;
  float wratio = (float)w/(wtemplate);

  std::cout << "Rotation in radians: " << angle << std::endl;

  int dx = roundToInt(wratio*(cs*referenceP[0] - sn*referenceP[1]));
  int dy = roundToInt(wratio*(sn*referenceP[0] + cs*referenceP[1]));
  // int x = referenceP[0] - dx;
  // int y = referenceP[1] - dy;
  // showImage.at<Vec3b>(dx, dy) = Vec3b(0, 0, 255);
  std::cout << "Position: " << dx << " " << dy << std::endl;
  std::cout << "Scale: " << wmin + (id_max[3]*rangeS) << std::endl;

  for (std::vector<std::vector<cv::Vec2i>>::size_type ii = 0; ii < Rtable.size(); ++ii){
    for (std::vector<cv::Vec2i>::size_type jj= 0; jj < Rtable[ii].size(); ++jj){
      int dx = roundToInt(wratio*(cs*Rtable[ii][jj][0] - sn*Rtable[ii][jj][1]));
      int dy = roundToInt(wratio*(sn*Rtable[ii][jj][0] + cs*Rtable[ii][jj][1]));
      int x = referenceP[0] - dx;
      int y = referenceP[1] - dy;
      if ( (x<nc)&&(y<nl)&&(x>-1)&&(y>-1) ){
        showImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
      }
    }
  }
  cv::imwrite("./result.png", showImage);
  return angle;
}

// load vector pts with all points from the contour
void Hough::readPoints(const cv::Mat& original_img, const cv::Mat& contour_img){

  cv::Mat template_img = contour_img.clone();
  // read original template image and its worked-out contour
  if(template_img.type() < 6) {
    cvtColor(template_img, template_img, cv::COLOR_GRAY2BGR);
  }
  cv::Mat input_img_gray;
  input_img_gray.create( cv::Size(original_img.cols, original_img.rows), CV_8UC1);
  if(original_img.type() > 6) {
    cvtColor(original_img, input_img_gray, cv::COLOR_BGR2GRAY);
  }
  //Mat template_img = imread("files\\contour_def.bmp", 1);

  // find reference point inside contour image and save it in variable refPoint
  int nl= template_img.rows;
  int nc= template_img.cols;
  // refPoint = Vec2i(template_img.cols/2,template_img.rows/2);
  cv::Vec2i refPoint = cv::Vec2i(0,0);

  // get Scharr matrices from original template image to obtain contour gradients
  cv::Mat dx;
  dx.create(cv::Size(original_img.cols, original_img.rows), CV_16SC1);
  cv::Sobel(input_img_gray, dx, CV_16S, 1, 0, cv::FILTER_SCHARR);
  cv::Mat dy;
  dy.create(cv::Size(original_img.cols, original_img.rows), CV_16SC1);
  cv::Sobel(input_img_gray, dy, CV_16S, 0, 1, cv::FILTER_SCHARR);
  // load points on vector
  pts.clear();
  int mindx = INT_MAX;
  int maxdx = INT_MIN;

  for (int j=0; j<nl; ++j) {
    cv::Vec3b* data= (cv::Vec3b*)(template_img.data + template_img.step.p[0]*j);
    for (int i=0; i<nc; ++i) {
      if (data[i]==cv::Vec3b(255, 255, 255))
      {
        short vx = dx.at<short>(j,i);
        short vy = dy.at<short>(j,i);
        Rpoint rpt;
        rpt.dx = refPoint(0)-i;
        rpt.dy = refPoint(1)-j;
        float a = atan2((float)vy, (float)vx); //radians
        rpt.phi = ((a > 0) ? a-PI/2 : a+PI/2);
        // update further right and left dx
        if (rpt.dx < mindx) mindx=rpt.dx;
        if (rpt.dx > maxdx) maxdx=rpt.dx;
        pts.push_back( rpt );
      }
    }
  }
  // maximum width of the contour
  wtemplate = maxdx-mindx+1;
}

// create Rtable from contour points
void Hough::readRtable(){
  // Setup Rtable
  Rtable.clear();
  Rtable.resize(intervals);

  // put points in the right interval, according to discretized angle and range size
  float range = PI/intervals;
  for (std::vector<Rpoint>::size_type t = 0; t < pts.size(); ++t){
    int angleindex = (int)((pts[t].phi+PI/2)/range);
    if (angleindex == intervals) angleindex=intervals-1;
    Rtable[angleindex].push_back(cv::Vec2i(pts[t].dx, pts[t].dy));
  }
}

float Hough::matchTemplate(const cv::Mat& searchImage, const cv::Mat& templateImage, const std::vector<std::vector<int>>& colorRange, const int expectedSize=0) {
  // Create a workcopy
  cv::Mat searchImageCopy = searchImage.clone();
  cv::Mat templateImageCopy = templateImage.clone();

  // For time measuring
  cv::TickMeter tm;

  // Set size of anticipated template in search image including small offset
  // Hough::wmin = expectedSize-(expectedSize*0.1);
  // Hough::wmax = expectedSize == 0 ? std::min(searchImageCopy.cols, searchImageCopy.rows) : expectedSize+(expectedSize*0.1);
  Hough::wmin = 20;
  Hough::wmax = 200;
  cv::Mat dummyCopy;
  // In case the input image is already binarized
  if(searchImageCopy.type() > 6) {
    dummyCopy = Image::binaryFromRange(searchImageCopy, colorRange);
    // cv::cvtColor(dummyCopy, dummyCopy, cv::COLOR_GRAY2BGR);
  }

  // Use morphology to clean the image
  // cv::Mat kernel = cv::Mat::ones( 8, 8, CV_32F );
  // cv::morphologyEx( searchImageCopy, searchImageCopy, cv::MORPH_OPEN, kernel );
  // cv::morphologyEx( searchImageCopy, searchImageCopy, cv::MORPH_CLOSE, kernel );

  // cv::resize(searchImageCopy, searchImageCopy, cv::Size(200, 200));
  // cv::resize(templateImageCopy, templateImageCopy, cv::Size(200, 200));

  // cv::imwrite("./binarySearch.png", searchImageCopy);


  std::cout << "createRTable" << std::endl;
  // create the Rtable from template
  createRtable(templateImageCopy);

  std::cout << "accumulate" << std::endl;
  // match template with search image
  accumulate(dummyCopy);

  // Find best candidate and return orientation
  cv::Mat displayCopy = searchImage.clone();
  cv::resize(displayCopy, displayCopy, cv::Size(200, 200));
  return bestCandidate(displayCopy);
}

