#pragma once

#include "layers.h"
#include "helpers.h"

namespace Nets {
	const int CONV = 3;
	const int VALID_PAD = 0, SAME_PAD = 1;

	class ConvL : public LayerCRTP<ConvL> {
	private:
		double lrate = 0.6;

		int padding = SAME_PAD;
		int kernel_sz = 3, kernel_cnt = 8;
		std::vector<matrix>* kernels = new std::vector<matrix>;

		void Generate_Kernels();
		
		matrix Pad(const matrix& mat);
	public:
		ConvL() = default;

		ConvL(int kernel_sz_, int kernel_cnt_, int padding_, double lrate_, double(*Init_Random_)(int, int) = Default_Random);
		ConvL(const ConvL& org);

		~ConvL();

		int Padding() const;
		
		std::vector<matrix> Kernels() const;

		double Lrate() const;
		void Set_Lrate(double new_lrate) override;

		row_vector Forward(const row_vector& in, bool rec = 0) override;
		row_vector Backward(const row_vector& grads) override;

		std::istream& Read(std::istream& stream) override;
		std::ostream& Write(std::ostream& stream) override;
	};
}