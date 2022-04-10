#pragma once

#include "helpers.h"
#include "cells.h"
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

        typedef row_vector(*rvd_F_rvd)(const row_vector&);
        typedef matrix(*mat_F_rvd)(const row_vector&);
        typedef double (*dfii)(int, int);

        std::vector<row_vector>* cache = new std::vector<row_vector>;
    public:
        virtual ~Layer() {}
        virtual Layer* Clone() const = 0;

        virtual int Input_Size() const;
        virtual void Set_In_Size(int) {}

        virtual int Output_Size() const;
        virtual void Set_Out_Size(int) {}

        virtual Cells::Cell* Cell() const;

        virtual void Set_Size(const std::vector<int>&) {};

        void Set_Init_Func(dfii);
        dfii Init_Func() const;

        void Clear_Cache();

        virtual void Set_Lrate(double) {}
        virtual void Set_Bias_Lrate(double) {}

        virtual void Set_Functions(rvd_F_rvd, mat_F_rvd) {}

        virtual row_vector Forward(const row_vector&, bool) = 0;
        virtual row_vector Backward(const row_vector&) = 0;

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