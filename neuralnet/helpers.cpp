#include "pch.h"
#include "helpers.h"

namespace Nets {

    double Scale(double val, double mini1, double maxi1, double mini2, double maxi2) {
        return (val - mini1) / (maxi1 - mini1) * (maxi2 - mini2) + mini2;
    }

    double Sigmoid(double x) {
        return 1. / (1. + pow(EULER, -x));
    }
    double Sigmoid_Reverse(double y) {
        return log(y / (1. - y));
    }
    double Sigmoid_Deriv(double x) {
        return Sigmoid(x) * (1. - Sigmoid(x));
    }

    double ReLU(double x) {
        return std::max(0., x);
    }
    double ReLU_Deriv(double x) {
        return x > 0;
    }

    double ELU(double x) {
        if (x > 0) return x;

        return exp(x) - 1;
    }
    double ELU_Deriv(double x) {
        if (x > 0) return 1;

        return exp(x);
    }
    double Squared_Error_Deriv(double o, double t) { return o - t; }

    double Default_Random(int in, int out) {
        static std::random_device rd;
        std::mt19937 gen(rd());

        std::normal_distribution<double> normal(0, sqrt(1. / in));
        return normal(gen);
    }
}
