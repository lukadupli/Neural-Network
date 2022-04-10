#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "helpers.h"

#include "layers.h"

#include "dense_layer.h"
#include "act_layer.h"
#include "recurrent_layer.h"
#include "conv_layer.h"

namespace Nets
{
    class Neural_Net
    {
    private:
        std::vector<Layer*> layers = {};

        typedef row_vector(*rvd_F_rvd_rvd)(const row_vector&, const row_vector&);
        typedef double(*d_F_rvd_rvd)(const row_vector&, const row_vector&);
        typedef row_vector(*rvd_F_rvd)(const row_vector&);
        typedef matrix(*mat_F_rvd)(const row_vector&);
        
        d_F_rvd_rvd Loss_Func = nullptr;
        rvd_F_rvd_rvd Loss_Deriv = Sq_Loss_Deriv;

    public:
        Neural_Net() = default;
        Neural_Net(std::vector<Layer*> layers_, rvd_F_rvd_rvd Loss_Deriv_ = Sq_Loss_Deriv, d_F_rvd_rvd Loss_Func_ = nullptr);
        Neural_Net(const char* path, rvd_F_rvd_rvd Loss_Deriv_ = Sq_Loss_Deriv, d_F_rvd_rvd Loss_Func_ = nullptr);
        ~Neural_Net();

        Neural_Net(const Neural_Net& org);

        std::vector<Layer*> Layers();
        std::vector<Layer*> Layers_Copy() const;
        rvd_F_rvd_rvd Get_Loss_Deriv() const;
        d_F_rvd_rvd Get_Loss_Func() const;

        void Manage_In_Sizes(int input_size);
        void Manage_Out_Sizes(int output_size);

        void Universal_Lrate(double lrate);
        void Universal_Bias_Lrate(double bias_lrate);
        void Universal_Activation(rvd_F_rvd Act_Func, mat_F_rvd Act_Deriv);

        row_vector Query(row_vector input, bool rec = 0);
        row_vector Query(const std::vector<double>& input, bool rec = 0);

        row_vector Back_Query(row_vector grads);
        row_vector Back_Query(const std::vector<double>& grads);

        double Train(const row_vector& input, const row_vector& target, bool rec = 0);
        double Train(const std::vector<double>& input, const std::vector<double>& target, bool rec = 0);

        void Save(std::ostream& stream);
        void Save(const std::string& path);
        void Save(const char* path);

        void Load(std::istream& stream);
        void Load(const std::string& path);
        void Load(const char* path);
    };
}

std::istream& operator>>(std::istream& istr, Nets::Neural_Net& net);
std::ostream& operator<<(std::ostream& ostr, Nets::Neural_Net& net);