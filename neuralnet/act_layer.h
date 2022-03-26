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

        row_vector (*Act_Func)(const row_vector&) = Sigmoid;
        row_vector (*Act_Deriv)(const row_vector&) = Sigmoid_Deriv;
    public:
        ActL() = default;
        ActL(rvd_F_rvd Act_Func_, rvd_F_rvd Act_Deriv_);

        ActL(const ActL&);

        ~ActL();

        int Input_Size() const override;
        void Set_In_Size(int in) override;

        int Output_Size() const override;
        void Set_Out_Size(int out) override;

        void Set_Size(const std::vector<int>& sizes) override;

        void Set_Lrate(double new_lrate) override;

        rvd_F_rvd Get_Act_Func() const;
        rvd_F_rvd Get_Act_Deriv() const;
        void Set_Functions(rvd_F_rvd New_Act_Func, rvd_F_rvd New_Act_Deriv) override;

        row_vector Forward(row_vector input, bool rec = 0) override;
        row_vector Backward(row_vector gradients) override;

        std::istream& Read(std::istream& stream) override;
        std::ostream& Write(std::ostream& stream) override;
    };
}