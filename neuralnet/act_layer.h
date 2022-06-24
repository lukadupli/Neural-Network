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

        std::vector<row_vector>* cache = new std::vector<row_vector>;

        row_vector (*Act_Func)(const row_vector&) = Sigmoid;
        matrix (*Act_Deriv)(const row_vector&) = SigmoidDeriv;
    public:
        ActL() = default;
        ActL(rvd_F_rvd Act_Func_, mat_F_rvd Act_Deriv_);

        ActL(const ActL&);

        ~ActL();

        int Input_Size() const override;
        void Set_In_Size(int in) override;

        int Output_Size() const override;
        void Set_Out_Size(int out) override;

        void Set_Size(const std::vector<int>& sizes) override;

        void Set_Lrate(double new_lrate) override;

        rvd_F_rvd Get_Act_Func() const;
        mat_F_rvd Get_Act_Deriv() const;
        void Set_Functions(rvd_F_rvd New_Act_Func, mat_F_rvd New_Act_Deriv) override;

        row_vector Forward(const row_vector& input, bool rec = 0) override;
        row_vector Backward(const row_vector& gradients) override;

        std::istream& Read(std::istream& stream) override;
        std::ostream& Write(std::ostream& stream) override;
    };
}