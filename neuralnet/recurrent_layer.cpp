#include "pch.h"
#include "recurrent_layer.h"

namespace Nets {
	RecL::RecL(Cells::Cell* cell_, int out_type_) {
		cell = cell_->Clone();

		out_type = out_type_;

		input_sz = cell->Input_Size();
		output_sz = cell->Output_Size();
	}
	RecL::RecL(const RecL& org) {
		cell = org.Cell()->Clone();

		input_sz = org.Input_Size();
		output_sz = org.Output_Size();

		out_type = org.Out_Type();
	}

	RecL::~RecL() {
		delete cell;
	}

	int RecL::Input_Size() const { return input_sz; }
	void RecL::Set_In_Size(int input_sz_){
		cell->Set_In_Size(input_sz_);
		input_sz = input_sz_;
	}

	int RecL::Output_Size() const { return output_sz; }
	void RecL::Set_Out_Size(int output_sz_) {
		cell->Set_Out_Size(output_sz_);
		output_sz = output_sz_;
	}

	int RecL::Out_Type() const { return out_type; }

	Cells::Cell* RecL::Cell() const { return cell; }

	row_vector RecL::Forward(row_vector in, bool rec) {
		if (in.size() % input_sz) throw std::runtime_error("Recurrent layer: cannot split the input into blocks of specified input size (remainder > 0)\n");
		
		in_count = in.size() / input_sz;

		row_vector out(in_count * output_sz);

		cell->Reset_Hid();
		for (int i = 0; i < in_count; i++) {
			row_vector res = cell->Forward(in.segment(i * input_sz, input_sz));

			out.segment(i * output_sz, output_sz) = res;
		}

		if (out_type == END) return out.tail(output_sz);
		
		return out;
	}

	row_vector RecL::Backward(row_vector grads) {
		if ((out_type == BEGIN  && grads.size() != in_count * output_sz) || (out_type == END && grads.size() != output_sz)) throw std::runtime_error("Recurrent layer : rececived gradient list doesn't match specified size");

		row_vector real_grads(in_count * output_sz);
		real_grads.setZero();

		real_grads.tail(grads.size()) = grads;

		row_vector new_grads(in_count * input_sz);
		
		cell->Reset_Back_Hid();
		for (int i = in_count - 1; i >= 0; i--) {
			row_vector res = cell->Backward(real_grads.segment(i * output_sz, output_sz));

			if (i < in_count) new_grads.segment(i * input_sz, input_sz) = res;
		}

		return new_grads;
	}

	std::istream& RecL::Read(std::istream& stream) {
		stream >> input_sz >> output_sz >> out_type;

		int cell_type;
		stream >> cell_type;

		switch (cell_type) {
		case Cells::BASIC_CELL:
			cell = new Cells::Basic;
			break;
		default:
			throw std::runtime_error("Recurrent layer read : corrupted file\n");
		}

		stream >> cell;

		return stream;
	}

	std::ostream& RecL::Write(std::ostream& stream) {
		stream << REC << '\n' << input_sz << ' ' << output_sz << ' ' << out_type << '\n' << cell << '\n';

		return stream;
	}

}