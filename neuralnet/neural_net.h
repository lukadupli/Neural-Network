#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "layers.h"
#include "helpers.h"

namespace Nets
{
    class Neural_Net
    {
    private:
        std::vector<Layer*> layers;

        double (*Error_Part_Deriv)(double, double);

        typedef double(*dfdd)(double, double);
    public:
        Neural_Net(std::vector<Layer*> layers_, double (*Error_Part_Deriv_)(double, double) = Squared_Error_Deriv);
        Neural_Net(const char* path, double (*Error_Part_Deriv_)(double, double));
        ~Neural_Net();

        Neural_Net(const Neural_Net& org);

        std::vector<Layer*> Layers() const;
        dfdd Get_Error_Deriv() const;

        void Manage_In_Sizes(int input_size);
        void Manage_Out_Sizes(int output_size);

        void Universal_Lrate(double lrate);
        void Universal_Bias_Lrate(double bias_lrate);
        void Universal_Activation(double (*Act_Func)(double), double (*Act_Func_Deriv)(double));

        row_vector Query(row_vector input);
        row_vector Query(const std::vector<double>& input);

        row_vector Back_Query(row_vector grads);
        row_vector Back_Query(const std::vector<double>& grads);

        row_vector Train(const row_vector& input, const row_vector& target);
        row_vector Train(const std::vector<double>& input, const std::vector<double>& target);

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