#pragma once

#include "layers.h"

#include <iostream>
#include <vector>

namespace Nets
{
    const int ACT = 1;

    class ActL : public LayerCRTP<ActL>
    {
    private:
        int sz;

        double (*Act_Func)(double);
        double (*Act_Deriv)(double);

        typedef double(*dfd)(double);

        row_vector cache;
    public:
        ActL() = default;
        ActL(double (*Act_Func_)(double), double (*Act_Deriv_)(double));

        ActL(const ActL&);

        int Input_Size() const override;
        void Set_In_Size(int in) override;

        int Output_Size() const override;
        void Set_Out_Size(int out) override;

        void Set_Size(const std::vector<int>& sizes) override;

        void Set_Lrate(double new_lrate) override;

        dfd Get_Act_Func() const;
        dfd Get_Act_Deriv() const;
        void Set_Functions(double (*New_Act_Func)(double), double (*New_Act_Deriv)(double)) override;

        row_vector Forward(row_vector input) override;
        row_vector Backward(row_vector gradients) override;

        std::istream& Read(std::istream& stream) override;
        std::ostream& Write(std::ostream& stream) override;
    };
}