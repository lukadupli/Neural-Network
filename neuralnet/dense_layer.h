#pragma once

#include "layers.h"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <Eigen/Dense>

namespace Nets
{
    const int DENSE = 0;

    class DenseL : public LayerCRTP<DenseL>
    {
    private:
        double lrate, bias_lrate;

        int input_sz, output_sz;
        Eigen::MatrixXd* weights;
        double bias;

        void Weight_Init(int input, int output, bool del = 1);

    public:
        DenseL() = default;
        DenseL(int input_sz_, int output_sz_, double lrate_, double bias_lrate_ = NAN, double (*Init_Random_)(int, int) = Default_Random);
        DenseL(const DenseL&);

        ~DenseL();

        int Input_Size() const override;
        void Set_In_Size(int in) override;

        int Output_Size() const override;
        void Set_Out_Size(int out) override;

        void Set_Size(const std::vector<int>& sizes) override;

        double Lrate() const;
        void Set_Lrate(double new_lrate) override;

        double Bias_Lrate() const;
        void Set_Bias_Lrate(double new_bias_lrate) override;

        double Bias() const;
        Eigen::MatrixXd Weights() const;

        row_vector Forward(row_vector input, bool rec = 0) override;
        row_vector Backward(row_vector gradients) override;

        std::istream& Read(std::istream& stream) override;
        std::ostream& Write(std::ostream& stream) override;
    };
}