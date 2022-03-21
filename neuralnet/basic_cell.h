#pragma once

#include "cells.h"

namespace Nets::Cells {
    const int BASIC_CELL = 0;

    class Basic : public CellCRTP<Basic>
    {
    private:
        row_vector *hid = new row_vector, *back_hid = new row_vector;

        Neural_Net *gate;
    public:
        Basic() = default;

        Basic(int input_sz_, int hidden_sz_, int output_sz_, const Neural_Net& gate_);
        Basic(const Basic& org);
        ~Basic();

        row_vector& Hidden() const;
        row_vector& Back_Hid() const;
        Neural_Net& Gate() const override;

        void Reset_Hid() override;
        void Reset_Back_Hid() override;

        void Set_In_Size(int input_sz_) override;
        void Set_Out_Size(int output_sz_) override;

        row_vector Forward(row_vector in) override;
        row_vector Backward(row_vector grads) override;

        std::istream& Read(std::istream& stream) override;
        std::ostream& Write(std::ostream& stream) override;
    };
}