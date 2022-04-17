#include "pch.h"
#include "conv_layer.h"

namespace Nets {

	void ConvL::Generate_Kernels() {
		kernels->clear();

		for (int i = 0; i < kernel_cnt; i++) {
			kernels->push_back(matrix(kernel_sz, kernel_sz));
			for (int x = 0; x < kernel_sz; x++) {
				for (int y = 0; y < kernel_sz; y++) kernels->back()(x, y) = Init_Random(kernel_sz * kernel_sz, 1);
			}
		}
	}

	matrix ConvL::Pad(const matrix& mat) {
		if (padding == VALID_PAD) return mat;

		matrix ret = matrix::Zero(mat.rows() + kernel_sz - 1, mat.cols() + kernel_sz - 1);

		int row_start = (kernel_sz - 1) / 2 + ((kernel_sz - 1) % 2);
		int col_start = (kernel_sz - 1) / 2 + ((kernel_sz - 1) % 2);
		for (int i = 0; i < mat.rows(); i++) {
			for (int j = 0; j < mat.cols(); j++) ret(row_start + i, col_start + j) = mat(i, j);
		}

		return ret;
	}

	ConvL::ConvL(int kernel_sz_, int kernel_cnt_, int padding_, double lrate_, double(*Init_Random_)(int, int))
	{
		kernel_sz = kernel_sz_;
		kernel_cnt = kernel_cnt_;
		padding = padding_;
		lrate = lrate_;
		Init_Random = Init_Random_;

		Generate_Kernels();
	}

	ConvL::ConvL(const ConvL& org) {
		*kernels = org.Kernels();

		kernel_sz = kernels->front().rows();
		kernel_cnt = kernels->size();
		padding = org.Padding();

		lrate = org.Lrate();

		Init_Random = org.Init_Func();
	}

	ConvL::~ConvL() {
		delete kernels;
	}

	int ConvL::Padding() const { return padding; }
	std::vector<matrix> ConvL::Kernels() const { return *kernels; }

	double ConvL:: Lrate() const { return lrate; }
	void ConvL::Set_Lrate(double new_lrate) { lrate = new_lrate; }

	row_vector ConvL::Forward(const row_vector& in, bool rec) { return row_vector{}; }
	row_vector ConvL::Backward(const row_vector& grads) { return row_vector{}; }

	std::istream& ConvL::Read(std::istream& str) { return str; }
	std::ostream& ConvL::Write(std::ostream& str) { return str; }
}
