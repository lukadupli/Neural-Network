#pragma once

#include "layers.h"
#include "helpers.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace Nets {
	const int CONV = 3;

	template<typename Scalar>
	inline Eigen::Tensor<Scalar, 4> Convolve(const Eigen::Tensor<Scalar, 3>& input, const Eigen::Tensor<Scalar, 3>& kernel, Eigen::PaddingType padding = Eigen::PADDING_SAME) {
		int input_d = input.dimension(0), input_w = input.dimension(1), input_h = input.dimension(2);
		int kernel_w = kernel.dimension(0), kernel_h = kernel.dimension(1), kernel_d = kernel.dimension(2);

		int output_d = input_d;
		int output_w, output_h;
		if (padding == Eigen::PADDING_SAME) {
			output_w = input_w;
			output_h = input_h;
		}
		else if (padding == Eigen::PADDING_VALID) {
			output_w = input_w - kernel_w + 1;
			output_h = input_h - kernel_h + 1;
		}
		else throw std::runtime_error("Invalid padding value\n");

		return input
			.extract_image_patches(kernel_w, kernel_h, 1i64, 1i64, 1i64, 1i64, padding)
			.reshape(Eigen::array<ptrdiff_t, 2>{input_d * output_w * output_h, kernel_w * kernel_h})
			.contract(
				kernel.reshape(Eigen::array<ptrdiff_t, 2>{kernel_w * kernel_h, kernel_d}),
				Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>{1, 0}})
			.reshape(Eigen::array<ptrdiff_t, 4>{output_d, output_w, output_h, kernel_d});
	}

	class ConvL : public LayerCRTP<ConvL> {
	private:
		double lrate = 0.6;

		int kernel_sz = 3, kernel_cnt = 8;
		std::vector<matrix>* kernels = new std::vector<matrix>;

		std::vector<std::vector<matrix>>* cache = new std::vector<std::vector<matrix>>;

		void Generate_Kernels();
		
		matrix Pad(const matrix& mat);
	public:
		ConvL() = default;

		ConvL(int kernel_sz_, int kernel_cnt_, double lrate_, double(*Init_Random_)(int, int) = DefaultRandom);
		ConvL(const ConvL& org);

		~ConvL();
		
		std::vector<matrix> Kernels() const;

		double Lrate() const;
		void Set_Lrate(double new_lrate) override;

		row_vector Forward(const row_vector& in, bool rec = 0) override;
		row_vector Backward(const row_vector& grads) override;

		std::istream& Read(std::istream& stream) override;
		std::ostream& Write(std::ostream& stream) override;
	};
}