#pragma once

#include "cells.h"

///Basic cell

namespace Nets::Cells {
    const int BASIC_CELL = 0;

    class Basic : public CellCRTP<Basic>
    {
    private:
        row_vector* hid;

        Neural_Net* gate;
    public:
        Basic() = default;

        Basic(Neural_Net gate_);
        Basic(const Basic& org);
        ~Basic();

        row_vector& Hidden() const;
        Neural_Net& Gate() const;

        void Reset_Hid() override;

        row_vector Forward(row_vector in) override;
        row_vector Backward(row_vector grads) override;

        std::istream& Read(std::istream& stream) override;
        std::ostream& Write(std::ostream& stream) override;
    };
}