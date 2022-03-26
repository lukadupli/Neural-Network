#include "pch.h"
#include "basic_cell.h"

namespace Nets::Cells
{
    Basic::Basic(int input_sz_, int hidden_sz_, int output_sz_, const Neural_Net& gate_) {
        input_sz = input_sz_;
        hidden_sz = hidden_sz_;
        output_sz = output_sz_;

        hid = new row_vector(hidden_sz);
        hid->setZero();

        back_hid = new row_vector(hidden_sz);
        back_hid->setZero();
        
        gate = new Neural_Net(gate_);
        gate->Manage_In_Sizes(input_sz + hidden_sz);
        gate->Manage_Out_Sizes(output_sz + hidden_sz);
    }

    Basic::Basic(const Basic& org) {
        input_sz = org.Input_Size();
        hidden_sz = org.Hidden_Size();
        output_sz = org.Output_Size();

        hid = new row_vector(org.Hidden());
        back_hid = new row_vector(org.Back_Hid());

        gate = new Neural_Net(org.Gate());
    }

    Basic::~Basic() {
        delete hid, back_hid, gate;
    }

    row_vector& Basic::Hidden() const { return *hid; }
    row_vector& Basic::Back_Hid() const { return *back_hid; }
    Neural_Net& Basic::Gate() const { return *gate; }

    void Basic::Reset_Hid() { hid->setZero(); }
    void Basic::Reset_Back_Hid() { back_hid->setZero(); }

    void Basic::Set_In_Size(int input_sz_) {
        input_sz = input_sz_;
        gate->Manage_In_Sizes(input_sz + hidden_sz);
    }
    void Basic::Set_Out_Size(int output_sz_) {
        output_sz = output_sz_;
        gate->Manage_Out_Sizes(output_sz + hidden_sz);
    }

    row_vector Basic::Forward(row_vector in) {
        if(in.size() != input_sz) throw std::runtime_error("Basic cell: rececived query list doesn't match specified size\n");

        row_vector carry(input_sz + hidden_sz);
        carry << in, *hid;

        carry = gate->Query(carry, 1);

        *hid = carry.tail(hidden_sz);
        return carry.head(output_sz);
    }

    row_vector Basic::Backward(row_vector grads) {
        if (grads.size() != output_sz) throw std::runtime_error("Basic cell: rececived gradient list doesn't match specified size\n");

        row_vector carry(output_sz + hidden_sz);
        carry << grads, *back_hid;

        carry = gate->Back_Query(carry);

        *back_hid = carry.tail(hidden_sz);
        return carry.head(input_sz);
    }

    std::istream& Basic::Read(std::istream& stream) {
        delete gate;
        gate = new Neural_Net;

        stream >> input_sz >> hidden_sz >> output_sz >> *gate;

        delete hid;
        hid = new row_vector(hidden_sz);
        hid->setZero();

        delete back_hid;
        back_hid = new row_vector(hidden_sz);
        back_hid->setZero();

        return stream;
    }
    std::ostream& Basic::Write(std::ostream& stream) {
        stream << BASIC_CELL << '\n' << input_sz << ' ' << hidden_sz << ' ' << output_sz << '\n' << *gate;

        return stream;
    }
}
