#include "pch.h"
#include "gru_cell.h"

namespace Nets::Cells {

	GRU::GRU(int input_sz_, int hidden_sz_, int output_sz_) {
		input_sz = input_sz_;
		hidden_sz = hidden_sz_;
		output_sz = output_sz_;

		hid = new row_vector(hidden_sz);

		reset_gate = new Neural_Net({
			new DenseL(input_sz + hidden_sz, hidden_sz, 0.6, 1.2),
			new ActL(Sigmoid, Sigmoid_Deriv)
			});

		update_gate = new Neural_Net({
			new DenseL(input_sz + hidden_sz, hidden_sz, 0.6, 1.2),
			new ActL(Sigmoid, Sigmoid_Deriv)
			});

		output_gate = new Neural_Net({
			new DenseL(input_sz, hidden_sz, 0.01, 0.02),
			new ActL(Tanh, Tanh_Deriv)
			});


	}

	GRU::GRU(const GRU& org) {
		input_sz = org.Input_Size();
		hidden_sz = org.Hidden_Size();
		output_sz = org.Output_Size();

		hid = new row_vector(org.Hidden());

		reset_gate = new Neural_Net(org.Reset_Gate());
		update_gate = new Neural_Net(org.Update_Gate());
		output_gate = new Neural_Net(org.Output_Gate());
	}

	GRU::~GRU() {
		delete hid, update_gate, reset_gate, output_gate,
			resetr_cache, hid_cache, updatez_cache, updateh_cache;
	}

	row_vector& GRU::Hidden() const { return *hid; }

	Neural_Net& GRU::Update_Gate() const { return *update_gate; }
	Neural_Net& GRU::Reset_Gate() const { return *reset_gate; }
	Neural_Net& GRU::Output_Gate() const { return *output_gate; }

	void GRU::Reset_Hid() { hid->setZero(); hid_cache->clear(); }

	void GRU::Set_In_Size(int input_sz_) {
		input_sz = input_sz_;

		reset_gate->Manage_In_Sizes(input_sz + hidden_sz);
		update_gate->Manage_In_Sizes(input_sz + hidden_sz);
		output_gate->Manage_In_Sizes(input_sz);
	}

	void GRU::Set_Hid_Size(int hidden_sz_) {
		hidden_sz = hidden_sz_;

		reset_gate->Manage_In_Sizes(input_sz + hidden_sz);
		update_gate->Manage_In_Sizes(input_sz + hidden_sz);
		output_gate->Manage_In_Sizes(input_sz);

		reset_gate->Manage_Out_Sizes(hidden_sz);
		update_gate->Manage_Out_Sizes(hidden_sz);
		output_gate->Manage_Out_Sizes(hidden_sz);
	}

	void GRU::Set_Out_Size(int output_sz_) {

	}

	row_vector GRU::Forward(const row_vector& in) {
		if (in.size() != input_sz) throw std::runtime_error("GRU cell: rececived query list doesn't match specified size\n");
		hid_cache->push_back(*hid);

		row_vector inhid_up(input_sz + hidden_sz);
		inhid_up << in, *hid;

		row_vector inhid_down(input_sz + hidden_sz);
		row_vector temp = reset_gate->Query(inhid_up, 1);
		resetr_cache->push_back(temp);
		inhid_down << in, hid->cwiseProduct(temp);

		inhid_up = update_gate->Query(inhid_up, 1);
		updatez_cache->push_back(inhid_up);

		row_vector update = row_vector::Ones(inhid_up.size()) - inhid_up;

		*hid = hid->cwiseProduct(update);

		inhid_down = output_gate->Query(inhid_down, 1);
		updateh_cache->push_back(inhid_down);
		inhid_down = inhid_down.cwiseProduct(inhid_up);

		*hid += inhid_down;

		return *hid;
	}

	row_vector GRU::Backward(const row_vector& grads) {
		if (grads.size() != hidden_sz) throw std::runtime_error("GRU cell: rececived gradient list doesn't match specified size\n");
		if (hid_cache->empty()) throw std::runtime_error("GRU cell: backward without previous forward\n");

		row_vector prev_hid = hid_cache->back(); hid_cache->pop_back();
		row_vector resetr = resetr_cache->back(); resetr_cache->pop_back();
		row_vector updatez = updatez_cache->back(); updatez_cache->pop_back();
		row_vector updateh = updateh_cache->back(); updateh_cache->pop_back();

		row_vector inhid_down = output_gate->Back_Query(grads.cwiseProduct(updatez));

		updatez = row_vector::Ones(updatez.size()) - updatez;

		

	}

	std::istream& GRU::Read(std::istream& stream) {
		stream >> input_sz >> hidden_sz >> *reset_gate >> *update_gate >> *output_gate;

		return stream;
	}

	std::ostream& GRU::Write(std::ostream& stream) {
		stream << GRU_CELL << '\n' << input_sz << ' ' << hidden_sz << '\n';
		stream << *reset_gate << '\n' << *update_gate << '\n' << *output_gate << '\n';

		return stream;
	}
}
