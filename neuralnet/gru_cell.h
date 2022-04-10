#pragma once

#include "cells.h"

namespace Nets::Cells {
	const int GRU_CELL = 1;

	class GRU : public CellCRTP<GRU> {
	private:
		row_vector* hid = nullptr;

		std::vector<row_vector>* resetr_cache = new std::vector<row_vector>;
		std::vector<row_vector>* hid_cache = new std::vector<row_vector>;
		std::vector<row_vector>* updatez_cache = new std::vector<row_vector>;
		std::vector<row_vector>* updateh_cache = new std::vector<row_vector>;

		Neural_Net* update_gate = nullptr;
		Neural_Net* reset_gate = nullptr;
		Neural_Net* output_gate = nullptr;

	public:
		GRU();

		GRU(int input_sz_, int hidden_sz_);
		GRU(const GRU& org);
		~GRU();

		row_vector& Hidden() const;
		Neural_Net& Update_Gate() const override;
		Neural_Net& Reset_Gate() const override;
		Neural_Net& Output_Gate() const override;

		void Reset_Hid(bool fwd) override;

		void Set_In_Size(int input_sz_) override;
		void Set_Hid_Size(int hidden_sz_) override;

		row_vector Forward(const row_vector& in) override;
		row_vector Backward(const row_vector& grads) override;

		std::istream& Read(std::istream& stream) override;
		std::ostream& Write(std::ostream& stream) override;
	};
}