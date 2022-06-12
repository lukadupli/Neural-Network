#include "pch.h"
#include "act_layer.h"

namespace Nets
{
    ActL::ActL(rvd_F_rvd Act_Func_, mat_F_rvd Act_Deriv_) {
        Act_Func = Act_Func_;
        Act_Deriv = Act_Deriv_;
    }
    ActL::ActL(const ActL& org) {
        sz = org.Input_Size();

        Act_Func = org.Get_Act_Func();
        Act_Deriv = org.Get_Act_Deriv();
    }

    ActL::~ActL(){
        delete cache;
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
    ActL::mat_F_rvd ActL::Get_Act_Deriv() const { return Act_Deriv; }
    void ActL::Set_Functions(rvd_F_rvd New_Act_Func, mat_F_rvd New_Act_Deriv) {
        Act_Func = New_Act_Func;
        Act_Deriv = New_Act_Deriv;
    }

    row_vector ActL::Forward(const row_vector& input, bool rec) { 
        if (!rec) cache->clear();

        cache->push_back(input);
        return Act_Func(input); 
    }

    row_vector ActL::Backward(const row_vector& gradients) {
        if (cache->empty()) throw std::runtime_error("Backward without previous forward\n");

        matrix deriv = Act_Deriv(cache->back());
        cache->pop_back();

        return (deriv * gradients.transpose()).transpose();
    }

    std::istream& ActL::Read(std::istream& stream) { stream >> Act_Func >> Act_Deriv; return stream; }
    std::ostream& ActL::Write(std::ostream& stream) {
        stream << ACT << '\n' << Act_Func << ' ' << Act_Deriv << '\n';
        return stream;
    }
}
