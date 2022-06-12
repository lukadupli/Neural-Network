#pragma once

#include <vector>
#include <map>
#include <algorithm>
#include <random>

#include <Eigen/Dense>

namespace Nets {

    const double EULER = exp(1.);

    const int SAME_SIZE = -1;
    const double SAME_DBL = NAN;
    const std::vector<int> SAME_VEC = { -1 };

    template <typename T> using ilist = std::initializer_list<T>;

    typedef Eigen::MatrixXd matrix;
    typedef Eigen::RowVectorXd row_vector;
    typedef Eigen::VectorXd col_vector;

    double Scale(double val, double mini1, double maxi1, double mini2, double maxi2);
    row_vector Clip(const row_vector& rv, double mini, double maxi);

    row_vector Sigmoid(const row_vector& x);
    matrix Sigmoid_Deriv(const row_vector& x);

    row_vector Tanh(const row_vector& x);
    matrix Tanh_Deriv(const row_vector& x);

    row_vector ReLU(const row_vector& x);
    matrix ReLU_Deriv(const row_vector& x);

    row_vector Softmax(const row_vector& in);
    matrix Softmax_Deriv(const row_vector& in);

    double Default_Random(int, int);

    double Cross_Entropy_Loss(const row_vector& out, const row_vector& target);
    row_vector Cross_Entropy_Loss_Deriv(const row_vector& out, const row_vector& target);

    row_vector Sq_Loss_Deriv(const row_vector& out, const row_vector& target);
    double Sq_Loss(const row_vector& out, const row_vector& target);

    matrix RowVec2Matrix(const row_vector& rv, int x, int y);
    row_vector Matrix2RowVec(const matrix& mat);

    // --------------- Act --------------- //
    typedef row_vector(*rvd_F_rvd)(const row_vector&);

    std::vector<rvd_F_rvd> ActDecode{ Sigmoid, Tanh, ReLU, Softmax };
    std::map<rvd_F_rvd, int> ActEncode{
        {Sigmoid, 0},
        {Tanh, 1},
        {ReLU, 2},
        {Softmax, 3}
    };

    std::istream& operator>>(std::istream& str, rvd_F_rvd& func);
    std::ostream& operator<<(std::ostream& str, rvd_F_rvd& func);

    // --------------- ActDeriv --------------- //
    typedef matrix(*mat_F_rvd)(const row_vector&);

    std::vector<mat_F_rvd> ActDerivDecode{ Sigmoid_Deriv, Tanh_Deriv, ReLU_Deriv, Softmax_Deriv };
    std::map<mat_F_rvd, int> ActDerivEncode{
        {Sigmoid_Deriv, 0},
        {Tanh_Deriv, 1},
        {ReLU_Deriv, 2},
        {Softmax_Deriv, 3}
    };

    std::istream& operator>>(std::istream& str, mat_F_rvd& func);
    std::ostream& operator<<(std::ostream& str, mat_F_rvd& func);
    
    // --------------- Loss --------------- //
    typedef double(*d_F_rvd_rvd)(const row_vector&, const row_vector&);

    std::vector<d_F_rvd_rvd> LossDecode{ Sq_Loss, Cross_Entropy_Loss};
    std::map<d_F_rvd_rvd, int> LossEncode{
        {Sq_Loss, 0},
        {Cross_Entropy_Loss, 1}
    };

    std::istream& operator>>(std::istream& str, d_F_rvd_rvd& func);
    std::ostream& operator<<(std::ostream& str, d_F_rvd_rvd& func);

    // --------------- LossDeriv --------------- //
    typedef row_vector(*rvd_F_rvd_rvd)(const row_vector&, const row_vector&);

    std::vector<rvd_F_rvd_rvd> LossDerivDecode{ Sq_Loss_Deriv, Cross_Entropy_Loss_Deriv };
    std::map<rvd_F_rvd_rvd, int> LossDerivEncode{
        {Sq_Loss_Deriv, 0},
        {Cross_Entropy_Loss_Deriv, 1}
    };

    std::istream& operator>>(std::istream& str, rvd_F_rvd_rvd& func);
    std::ostream& operator<<(std::ostream& str, rvd_F_rvd_rvd& func);

    // ----------------- END ----------------- //

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