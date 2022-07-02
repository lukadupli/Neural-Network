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
		matrix ret = matrix::Zero(mat.rows() + kernel_sz - 1, mat.cols() + kernel_sz - 1);

		int row_start = (kernel_sz - 1) / 2 + ((kernel_sz - 1) % 2);
		int col_start = (kernel_sz - 1) / 2 + ((kernel_sz - 1) % 2);

		ret.block(row_start, col_start, mat.rows(), mat.cols()) = mat;

		return ret;
	}

	ConvL::ConvL(int kernel_sz_, int kernel_cnt_, double lrate_, double(*Init_Random_)(int, int))
	{
		kernel_sz = kernel_sz_;
		kernel_cnt = kernel_cnt_;
		lrate = lrate_;
		Init_Random = Init_Random_;

		Generate_Kernels();
	}

	ConvL::ConvL(const ConvL& org) {
		*kernels = org.Kernels();

		kernel_sz = kernels->front().rows();
		kernel_cnt = kernels->size();

		lrate = org.Lrate();

		Init_Random = org.Init_Func();
	}

	ConvL::~ConvL() {
		delete cache, kernels;
	}

	std::vector<matrix> ConvL::Kernels() const { return *kernels; }

	double ConvL::Lrate() const { return lrate; }
	void ConvL::Set_Lrate(double new_lrate) { lrate = new_lrate; }

	row_vector ConvL::Forward(const row_vector& in, bool rec) { 
		std::vector<matrix> real_in = RowVecTo3D(in);

		if (!rec) cache->clear();
		cache->push_back(real_in);

		std::vector<matrix> ret;
		
		for (auto& mat : real_in) {
			mat = Pad(mat);

			for (int x = 0; x < kernel_cnt; x++) {

				ret.push_back(matrix{ mat.rows() - kernel_sz + 1, mat.cols() - kernel_sz + 1 });
				for (int i = 0; i < mat.rows() - kernel_sz + 1; i++) {
					for (int j = 0; j < mat.cols() - kernel_sz + 1; j++) {
						ret.back()(i, j) =
							mat.block(i, j, kernel_sz, kernel_sz).cwiseProduct((*kernels)[x]).sum();
					}
				}
			}
		}

		return ThreeDToRowVec(ret);
	}
	row_vector ConvL::Backward(const row_vector& grads) { 
		if (cache->empty()) throw std::runtime_error("Backward without previous forward\n");
		std::vector<matrix> real_grads = RowVecTo3D(grads), ret;
		
		int rows = real_grads[0].rows(), cols = real_grads[0].cols();

		// new grads
		for (int i = 0; i < real_grads.size(); i++) {
			if(i % kernel_cnt == 0) ret.push_back(matrix{ rows, cols });

			real_grads[i] = Pad(real_grads[i]);
			for (int x = 0; x < rows; x++) {
				for (int y = 0; y < cols; y++) {
					ret.back()(x, y) = real_grads[i].block(x, y, kernel_sz, kernel_sz).cwiseProduct((*kernels)[i % kernel_cnt]).sum();
				}
			}
		}

		// kernel updates
		for (int k = 0; k < kernel_cnt; k++) {
			for (int g = k; g < real_grads.size(); g += kernel_cnt) {
				matrix cachenow = Pad(cache->back()[g / kernel_cnt]);

				for (int x = 0; x < rows; x++) {
					for (int y = 0; y < cols; y++) {
						(*kernels)[k] -= lrate * real_grads[g].block(x, y, kernel_sz, kernel_sz).cwiseProduct(
						cachenow.block(x, y, kernel_sz, kernel_sz));
					}
				}
			}
		}

		cache->pop_back();
		return ThreeDToRowVec(ret);
	}

	std::istream& ConvL::Read(std::istream& str) { 
		str >> kernel_sz >> kernel_cnt >> lrate;
		kernels->clear();

		for (int i = 0; i < kernel_cnt; i++) {
			kernels->push_back(matrix{ kernel_sz, kernel_sz });
			
			for (int x = 0; x < kernel_sz; x++) {
				for (int y = 0; y < kernel_sz; y++)
					str >> kernels->back()(x, y);
			}
		}

		return str; 
	}

	std::ostream& ConvL::Write(std::ostream& str) { 
		str << CONV << '\n' << kernel_sz << ' ' << kernel_cnt << ' ' << lrate << '\n';
		for (auto& e : *kernels) {
			str << e << '\n';
		}

		return str; 
	}
}
