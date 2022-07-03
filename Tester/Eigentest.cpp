#define EXCLUD
#ifndef EXCLUDE

#include <iostream>
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../neuralnet/neural_net.h"

using namespace std;

template<typename T, int dims> void print(Eigen::Tensor<T, dims>& tensor, bool flatten_2d = true, int d = 0, Eigen::array<ptrdiff_t, dims>& arr = *(new Eigen::array<ptrdiff_t, dims>)) {
	if (d == dims) {
		cout << tensor(arr);
		if (arr[dims - 1] < tensor.dimension(dims - 1) - 1) cout << ", ";
		return;
	}

	for (int i = 0; i < tensor.dimension(d); i++) {
		arr[d] = i;
		if (d < dims - 1) {
			if (i > 0) for (int _ = 0; _ < d; _++) cout << ' ';
			cout << '[';
		}

		print(tensor, flatten_2d, d + 1, arr);

		if (d < dims - 1) {
			cout << ']';
			if (flatten_2d && d == dims - 2) {
				if (i < tensor.dimension(d) - 1) cout << ", ";
			}
			else if (i < tensor.dimension(d) - 1) cout << '\n';
		}
	}
}

template<typename T, int dims> void print(Eigen::TensorMap<Eigen::Tensor<double, dims>>& tensor, bool flatten_2d, int d = 0, Eigen::array<ptrdiff_t, dims>& arr = *(new Eigen::array<ptrdiff_t, dims>)) {
	if (d == dims) {
		cout << tensor(arr);
		if (arr[dims - 1] < tensor.dimension(dims - 1) - 1) cout << ", ";
		return;
	}

	for (int i = 0; i < tensor.dimension(d); i++) {
		arr[d] = i;
		if (d < dims - 1) {
			if (i > 0) for (int _ = 0; _ < d; _++) cout << ' ';
			cout << '[';
		}

		print<double, dims>(tensor, flatten_2d, d + 1, arr);

		if (d < dims - 1) {
			cout << ']';
			if (flatten_2d && d == dims - 2) {
				if (i < tensor.dimension(d) - 1) cout << ", ";
			}
			else if (i < tensor.dimension(d) - 1) cout << '\n';
		}
	}
}

template<typename T, int dims> void printrev(Eigen::Tensor<T, dims>& tensor, int d = 0, Eigen::array<ptrdiff_t, dims>& arr = *(new Eigen::array<ptrdiff_t, dims>)) {
	if (d == dims) {
		cout << tensor(arr);
		if (arr[0] < tensor.dimension(0) - 1) cout << ", ";

		return;
	}

	for (int i = 0; i < tensor.dimension(dims - 1 - d); i++) {
		arr[dims - 1 - d] = i;
		if (d < dims - 1) {
			if (i > 0) for (int _ = 0; _ < d; _++) cout << ' ';
			cout << '[';
		}

		printrev(tensor, d + 1, arr);

		if (d < dims - 1) {
			cout << ']';
			if (i < tensor.dimension(dims - 1 - d) - 1) cout << '\n';
		}
	}
}

int input_r = 3, input_c = 3, input_d = 2;
int kernel_r = 2, kernel_c = 2, kernel_d = 2;

Eigen::Tensor<double, 3> in(input_d, input_r, input_c);
Eigen::Tensor<double, 3> kernel(kernel_r, kernel_c, kernel_d);

int main() {
	in.setValues({
		{{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9}},

		{{1, 1, 1},
		{1, 0, 0},
		{0, 0, 1}}
		});
	kernel.setValues({
		{{1, 1}, {1, 0}},
		{{1, 1}, {1, 1}}
		});

	auto temp = Nets::Tensor3DToRowVec(in);
	auto temp2 = Nets::RowVecToTensor3D(temp);
	print(temp2, false);
	
	Eigen::RowVectorXd rvin(2 * 3 * 3 + 3);
	rvin << 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 2, 3, 3;

	cout << "\n\n";

	auto res = Nets::Convolve(in, kernel, Eigen::PADDING_SAME);
	print(res, false);

	/*cout << "Input tensor:\n";
	print(in);
	cout << "\n\n";

	cout << "Image patches:\n";
	Eigen::Tensor<double, 4> patches = in.extract_image_patches(kernel_r, kernel_c);
	print(patches, true);
	Eigen::Tensor<double, 4> psh = patches.shuffle(Eigen::array<ptrdiff_t, 4>{3, 0, 1, 2});
	cout << '\n';
	print(psh, false);
	cout << "\n\n";

	long long patches_total = 1;
	for (int i = 0; i < patches.NumDimensions; i++) {
		patches_total *= patches.dimension(i);
		cout << patches.dimension(i) << ' ';
	}
	cout << "\n\n";

	cout << "Reshaped patches:\n";
	Eigen::Tensor<double, 2> reshaped1 = psh.reshape(Eigen::array<ptrdiff_t, 2>({ patches_total / (kernel_r * kernel_c), kernel_r * kernel_c}));
	print(reshaped1, false);
	cout << "\n\n";

	cout << "Kernel:\n";
	print(kernel);
	cout << "\n\n";

	cout << "Reshaped kernel:\n";
	Eigen::Tensor<double, 2> reshaped_k = kernel.reshape(Eigen::array<ptrdiff_t, 2>({ kernel_r * kernel_c, kernel_d }));
	print(reshaped_k, false);
	cout << "\n\n";

	cout << "Contraction reshaped patches - reshaped kernel:\n";
	Eigen::Tensor<double, 2> contraction = reshaped1.contract(reshaped_k, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>{1, 0}});
	print(contraction, false);
	cout << "\n\n";

	cout << "Final reshaping:\n";
	Eigen::Tensor<double, 3> out = contraction.shuffle(Eigen::array<ptrdiff_t, 2>{1, 0}).reshape(Eigen::array<ptrdiff_t, 4>{input_d, input_r, input_c, kernel_d})
		.shuffle(Eigen::array<ptrdiff_t, 4>{0, 3, 1, 2}).reshape(Eigen::array<ptrdiff_t, 3>{input_d* kernel_d, input_r, input_c});

	print(out, false);
	cout << "\n\n";*/

	return 0;
}

#endif