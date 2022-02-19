#pragma once

#include <vector>
#include <algorithm>
#include <random>

#include <Eigen/Dense>

namespace Nets {

    const double EULER = exp(1.);

    const int SAME_SIZE = -1;
    const double SAME_DBL = NAN;
    const std::vector<int> SAME_VEC = { -1 };

    template <typename T> using ilist = std::initializer_list<T>;

    typedef Eigen::MatrixX <double> Matrix;
    typedef Eigen::RowVectorXd row_vector;
    typedef Eigen::VectorXd col_vector;

    double Scale(double val, double mini1, double maxi1, double mini2, double maxi2);

    double Sigmoid(double x);
    double Sigmoid_Reverse(double y);
    double Sigmoid_Deriv(double x);

    double ReLU(double x);
    double ReLU_Deriv(double x);

    double ELU(double x);
    double ELU_Deriv(double x);

    double Default_Random(int, int);

    double Squared_Error_Deriv(double o, double t);

    template <typename T>
    inline Eigen::RowVectorX <T> Vec2Eig(std::vector<T> v) {
        Eigen::RowVectorX <T> ret;
        ret.resize(v.size());

        for (int i = 0; i < v.size(); i++) {
            ret.coeffRef(i) = v[i];
        }

        return ret;
    }

}