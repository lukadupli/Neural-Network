#pragma once

#include <iostream>
#include "helpers.h"
#include "neural_net.h"

namespace Nets::Cells
{
    class Cell
    {
    protected:
        int input_sz, output_sz;
    public:
        virtual ~Cell() {}
        virtual Cell* Clone() const = 0;

        virtual void Reset_Hid() = 0;

        int Input_Size() const;
        int Output_Size() const;

        virtual row_vector Forward(row_vector in) = 0;
        virtual row_vector Backward(row_vector grads) = 0;

        virtual std::istream& Read(std::istream& stream) = 0;
        virtual std::ostream& Write(std::ostream& stream) = 0;
    };

    template<typename CType> class CellCRTP : public Cell
    {
    public:
        Cell* Clone() const override {
            return new CType(static_cast<const CType&>(*this));
        }
    };
}

#include "basic_cell.h"

std::istream& operator>>(std::istream& str, Nets::Cells::Cell* c);
std::ostream& operator<<(std::ostream& str, Nets::Cells::Cell* c);