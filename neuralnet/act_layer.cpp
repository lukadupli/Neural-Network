#include "pch.h"
#include "act_layer.h"

namespace Nets
{
    ActL::ActL(rvd_F_rvd Act_Func_, rvd_F_rvd Act_Deriv_) {
        Act_Func = Act_Func_;
        Act_Deriv = Act_Deriv_;
    }
    ActL::ActL(const ActL& org) {
        sz = org.Input_Size();

        Act_Func = org.Get_Act_Func();
        Act_Deriv = org.Get_Act_Deriv();
    }

    int ActL::Input_Size() const { return sz; };
    void ActL::Set_In_Size(int in) { Set_Size({ in }); }

    int ActL::Output_Size() const { return sz; };
    void ActL::Set_Out_Size(int out) { Set_Size({ out }); }

    void ActL::Set_Size(const std::vector<int>& sizes) {
        if (sizes[0] > 0) sz = sizes[0];
    }

    void ActL::Set_Lrate(double new_lrate) {};

    ActL::rvd_F_rvd ActL::Get_Act_Func() const { return Act_Func; }
    ActL::rvd_F_rvd ActL::Get_Act_Deriv() const { return Act_Deriv; }
    void ActL::Set_Functions(rvd_F_rvd New_Act_Func, rvd_F_rvd New_Act_Deriv) {
        Act_Func = New_Act_Func;
        Act_Deriv = New_Act_Deriv;
    }

    row_vector ActL::Forward(row_vector input) { 
        cache = input;
        return Act_Func(input); 
    }

    row_vector ActL::Backward(row_vector gradients) {
        row_vector ret = Act_Deriv(cache);

        for (int i = 0; i < gradients.size(); i++) ret(i) *= gradients(i);

        return ret;
    }

    std::istream& ActL::Read(std::istream& stream) { return stream; }
    std::ostream& ActL::Write(std::ostream& stream) {
        stream << ACT << '\n';
        return stream;
    }
}
