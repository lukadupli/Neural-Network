#pragma once

#include "layers.h"
#include "basic_cell.h"
#include "gru_cell.h"

namespace Nets {
	const int REC = 2;

	const int BEGIN = 0, END = 1;

	class RecL : public LayerCRTP<RecL> {
	private:
		int input_sz = 0, output_sz = 0;

		Cells::Cell* cell = nullptr;

		int out_type = 0;

		int in_count = 0;
	public:
		RecL() = default;
		RecL(Cells::Cell* cell_, int out_type_);
		RecL(const RecL& org);

		~RecL();

		int Input_Size() const override;
		void Set_In_Size(int input_sz) override;

		int Output_Size() const override;
		void Set_Out_Size(int input_sz) override;

		int Out_Type() const;
		
		Cells::Cell* Cell() const override;

		row_vector Forward(const row_vector& in, bool rec = 0) override;
		row_vector Backward(const row_vector& grads) override;

		std::istream& Read(std::istream& stream) override;
		std::ostream& Write(std::ostream& stream) override;
	};
}