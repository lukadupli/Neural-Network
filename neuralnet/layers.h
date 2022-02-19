#pragma once

#include "helpers.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <random>

#include <Eigen/Dense>

namespace Nets {
    class Layer
    {
    protected:
        double (*Init_Random)(int, int);

        typedef double (*dfii)(int, int);
    public:
        virtual ~Layer() {}
        virtual Layer* Clone() const = 0;

        virtual int Input_Size() const = 0;
        virtual void Set_In_Size(int) = 0;

        virtual int Output_Size() const = 0;
        virtual void Set_Out_Size(int) = 0;

        virtual void Set_Size(const std::vector<int>&) = 0;

        void Set_Init_Func(dfii);
        dfii Init_Func() const;

        virtual void Set_Lrate(double) = 0;
        virtual void Set_Bias_Lrate(double) {}

        virtual void Set_Functions(double (*)(double), double (*)(double)) {}

        virtual row_vector Forward(row_vector) = 0;
        virtual row_vector Backward(row_vector) = 0;

        virtual std::istream& Read(std::istream&) = 0;
        virtual std::ostream& Write(std::ostream&) = 0;
    };

    template<typename LType> class LayerCRTP : public Layer
    {
    public:
        Layer* Clone() const override {
            return new LType(static_cast<const LType&>(*this));
        }
    };
}

std::istream& operator>>(std::istream& istr, Nets::Layer* lay);
std::ostream& operator<<(std::ostream& ostr, Nets::Layer* lay);

#include "dense_layer.h"
#include "act_layer.h"