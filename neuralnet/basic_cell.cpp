#include "pch.h"
#include "basic_cell.h"

namespace Nets::Cells
{
    Basic::Basic(Neural_Net gate_) {
        input_sz = gate_.Layers().front()->Input_Size();
        output_sz = gate_.Layers().back()->Output_Size();

        hid = new row_vector(input_sz);
        hid->setZero();

        gate = new Neural_Net(gate_);
        gate->Manage_In_Sizes(2 * input_sz);
        gate->Manage_Out_Sizes(output_sz + input_sz);
    }
    Basic::Basic(const Basic& org) {
        input_sz = org.Input_Size();
        output_sz = org.Output_Size();

        hid = new row_vector(org.Hidden());

        gate = new Neural_Net(org.Gate());
    }

    Basic::~Basic() {
        delete hid, gate;
    }

    row_vector& Basic::Hidden() const { return *hid; }
    Neural_Net& Basic::Gate() const { return *gate; }

    void Basic::Reset_Hid() {
        hid->setZero();
    }

    row_vector Basic::Forward(row_vector in) {
        if(in.size() != 2 * input_sz) throw std::runtime_error("Basic cell: rececived query list doesn't match specified size\n");

        row_vector carry(in.size() + hid->size());
        carry << in, * hid;

        carry = gate->Query(carry);

        *hid = carry.tail(hid->size());
        return carry.head(input_sz);
    }

    row_vector Basic::Backward(row_vector grads) {
        if (grads.size() != input_sz + output_sz) throw std::runtime_error("Basic cell: rececived gradient list doesn't match specified size\n");
        return gate->Back_Query(grads);
    }

    std::istream& Basic::Read(std::istream& stream) {
        stream >> input_sz >> *gate;

        return stream;
    }
    std::ostream& Basic::Write(std::ostream& stream) {
        stream << BASIC_CELL << '\n' << input_sz << '\n' << *gate;

        return stream;
    }
}
