
#include <iostream>
#include <filesystem>
#include "../neuralnet/neural_net.h"
#include "../neuralnet/cells.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#define f first
#define s second

constexpr auto LOCATION = "C:/Users/lukad/OneDrive/Dokumenti/VS_Projects/Neural_Network/neural_net_data/";

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace Nets;
using namespace Nets::Cells;

Cell *cell = new Basic(Neural_Net({ new DenseL(3, 3, 0.6, 1.2), new ActL(Sigmoid, Sigmoid_Deriv) }));

#ifndef NO_MAIN

int main() {
	Cell *cell2 = cell->Clone();

	cout << cell2 << '\n' << cell;

	return 0;
}

#endif
