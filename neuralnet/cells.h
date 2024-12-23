#pragma once

#include <iostream>
#include "helpers.h"
#include "neural_net.h"

namespace Nets {
    class Neural_Net;
}

namespace Nets::Cells
{
    class Cell
    {
    protected:
        int input_sz, hidden_sz;
    public:
        virtual ~Cell() {}
        virtual Cell* Clone() const = 0;

        virtual void Reset_Hid(bool fwd) = 0;

        int Input_Size() const;
        int Hidden_Size() const;
        
        virtual Neural_Net& Gate() const;
        virtual Neural_Net& Update_Gate() const;
        virtual Neural_Net& Reset_Gate() const;
        virtual Neural_Net& Output_Gate() const;

        virtual void Set_In_Size(int input_sz_) = 0;
        virtual void Set_Hid_Size(int hidden_sz_) = 0;

        virtual row_vector Forward(const row_vector& in) = 0;
        virtual row_vector Backward(const row_vector& grads) = 0;

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

std::istream& operator>>(std::istream& str, Nets::Cells::Cell* c);
std::ostream& operator<<(std::ostream& str, Nets::Cells::Cell* c);