#define EXCLUD
#ifndef EXCLUDE

#include <iostream>

#include <Eigen/Dense>

#include "../neuralnet/helpers.h"

using namespace std;
using namespace Nets;

matrix mat(2, 3);
matrix Pad(const matrix& mat, int kernel_sz) {
	matrix ret = matrix::Zero(mat.rows() + kernel_sz - 1, mat.cols() + kernel_sz - 1);

	int row_start = (kernel_sz - 1) / 2 + ((kernel_sz - 1) % 2);
	int col_start = (kernel_sz - 1) / 2 + ((kernel_sz - 1) % 2);
	for (int i = 0; i < mat.rows(); i++) {
		for (int j = 0; j < mat.cols(); j++) ret(row_start + i, col_start + j) = mat(i, j);
	}

	return ret;
}

int main() {
	mat << 1, 2, 3, 4, 5, 6;
	Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> arr(2, 3);
	arr << 1, 2, 3, 4, 5, 6;

	cout << Pad(mat, 5);

	return 0;
}

#endif