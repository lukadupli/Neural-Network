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

    typedef Eigen::RowVectorXd row_vector;
    typedef Eigen::VectorXd col_vector;

    double Scale(double val, double mini1, double maxi1, double mini2, double maxi2);

    row_vector Sigmoid(const row_vector& x);
    row_vector Sigmoid_Deriv(const row_vector& x);

    row_vector Tanh(const row_vector& x);
    row_vector Tanh_Deriv(const row_vector& x);

    row_vector ReLU(const row_vector& x);
    row_vector ReLU_Deriv(const row_vector& x); 
    
    row_vector Softmax(const row_vector& in);
    row_vector Softmax_Deriv(const row_vector& in);

    double Default_Random(int, int);

    double Cross_Entropy_Loss(const row_vector& out, const row_vector& target);
    row_vector Cross_Entropy_Loss_Deriv(const row_vector& out, const row_vector& target);

    row_vector Sq_Loss_Deriv(const row_vector& out, const row_vector& target);
    double Sq_Loss(const row_vector& out, const row_vector& target);

    template <typename T>
    inline Eigen::RowVectorX <T> Vec2Eig(const std::vector<T>& v) {
        Eigen::RowVectorX <T> ret;
        ret.resize(v.size());

        for (int i = 0; i < v.size(); i++) {
            ret.coeffRef(i) = v[i];
        }

        return ret;
    }

}