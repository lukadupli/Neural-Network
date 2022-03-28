#define EXCLUDE
#ifndef EXCLUDE

#include <iostream>

#include <Eigen/Dense>
#include "../neuralnet/helpers.h"

using namespace std;
using namespace Nets;

row_vector rv1(3), rv2(3);

int main() {
	rv1 << 1, 2, 3;
	rv2 << 4, 5, 6;

	cout << rv1.cwiseProduct(rv2) << '\n';
	cout << rv1 << '\n';

	rv1 = rv1.cwiseProduct(rv2);
	cout << -rv1;

	return 0;
}

#endif