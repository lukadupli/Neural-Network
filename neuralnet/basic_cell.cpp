#include "pch.h"
#include "basic_cell.h"

namespace Nets::Cells
{
    Basic::Basic(int input_sz_, int hidden_sz_, const Neural_Net& gate_) {
        input_sz = input_sz_;
        hidden_sz = hidden_sz_;

        hid = new row_vector(hidden_sz);
        hid->setZero();
        
        gate = new Neural_Net(gate_);
        gate->Manage_In_Sizes(input_sz + hidden_sz);
        gate->Manage_Out_Sizes(hidden_sz);
    }

    Basic::Basic(const Basic& org) {
        input_sz = org.Input_Size();
        hidden_sz = org.Hidden_Size();

        hid = new row_vector(org.Hidden());

        gate = new Neural_Net(org.Gate());
    }

    Basic::~Basic() {
        delete hid, gate;
    }

    row_vector& Basic::Hidden() const { return *hid; }
    Neural_Net& Basic::Gate() const { return *gate; }

    void Basic::Reset_Hid(bool fwd) { hid->setZero(); }

    void Basic::Set_In_Size(int input_sz_) {
        input_sz = input_sz_;
        gate->Manage_In_Sizes(input_sz + hidden_sz);
    }
    void Basic::Set_Hid_Size(int hidden_sz_) {
        hidden_sz = hidden_sz_;
        gate->Manage_In_Sizes(input_sz + hidden_sz);
        gate->Manage_Out_Sizes(hidden_sz);
    }

    row_vector Basic::Forward(const row_vector& in) {
        if(in.size() != input_sz) throw std::runtime_error("Basic cell: rececived query list doesn't match specified size\n");

        row_vector carry(input_sz + hidden_sz);
        carry << in, *hid;

        *hid = gate->Query(carry, 1);

        return *hid;
    }

    row_vector Basic::Backward(const row_vector& grads) {
        if (grads.size() != hidden_sz) throw std::runtime_error("Basic cell: rececived gradient list doesn't match specified size\n");

        row_vector carry = *hid + grads;
        carry = gate->Back_Query(carry);

        *hid = carry.tail(hidden_sz);
        return carry.head(input_sz);
    }

    std::istream& Basic::Read(std::istream& stream) {
        if (!gate) gate = new Neural_Net;
        stream >> input_sz >> hidden_sz;
        gate->Load(stream);

        hid->resize(hidden_sz);
        hid->setZero();

        return stream;
    }
    std::ostream& Basic::Write(std::ostream& stream) {
        stream << BASIC_CELL << '\n' << input_sz << ' ' << hidden_sz << '\n';
        gate->Save(stream);

        return stream;
    }
}
