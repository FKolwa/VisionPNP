#include "../include/image.h"
#include "../include/color.h"
#include <iostream>
using namespace std;
using namespace cv;

int main(int argc, const char* argv[])
{
  //-- BEGIN Gripper part - uncomment to run:
  vector <vector<int>> thresh;
  thresh = Color::getHSVColorRange("../images/gripper.png");

  float orientation = Image::matchTemplate("../images/tiny-on-gripper.png", "../images/template-output.png", thresh);
  cout << "Orientation:" << endl;
  cout << orientation << endl;
  //-- END Gripper part

  //-- BEGIN Tray part - uncomment to run:
  vector<int> center = Image::findShape("../images/tray_resistor.png");
  cout << "Position:" << endl;
  cout << center[0] << " " << center[1] << endl;
  //-- END Tray part
}