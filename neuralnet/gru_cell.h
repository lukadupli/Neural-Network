#pragma once

#include "cells.h"
#include "neural_net.h"

namespace Nets::Cells {
	class GRU : public CellCRTP<GRU> {
	private:
		row_vector* hid = new row_vector;

		Neural_Net* update_gate = nullptr;
		Neural_Net* reset_gate = nullptr;
		Neural_Net* output_gate = nullptr;

	public:
		GRU() = default;

	};
}