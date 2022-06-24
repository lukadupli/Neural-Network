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

    double DefaultRandom(int, int);

    std::vector<matrix> RowVecTo3D(const row_vector& rv);
    row_vector ThreeDToRowVec(const std::vector<matrix>& mat);

    // --------------- Act --------------- //
    typedef row_vector(*rvd_F_rvd)(const row_vector&);

    row_vector Sigmoid(const row_vector& x);
    row_vector Tanh(const row_vector& x);
    row_vector ReLU(const row_vector& x);
    row_vector Softmax(const row_vector& in);
    
    std::istream& operator>>(std::istream& str, rvd_F_rvd& func);
    std::ostream& operator<<(std::ostream& str, rvd_F_rvd& func);

    // --------------- ActDeriv --------------- //
    typedef matrix(*mat_F_rvd)(const row_vector&);

    matrix SigmoidDeriv(const row_vector& x);
    matrix TanhDeriv(const row_vector& x);
    matrix ReLUDeriv(const row_vector& x);
    matrix SoftmaxDeriv(const row_vector& in);

    std::istream& operator>>(std::istream& str, mat_F_rvd& func);
    std::ostream& operator<<(std::ostream& str, mat_F_rvd& func);
    
    // --------------- Loss --------------- //
    typedef double(*d_F_rvd_rvd)(const row_vector&, const row_vector&);

    double SqLoss(const row_vector& out, const row_vector& target);
    double CrossEntropyLoss(const row_vector& out, const row_vector& target);

    std::istream& operator>>(std::istream& str, d_F_rvd_rvd& func);
    std::ostream& operator<<(std::ostream& str, d_F_rvd_rvd& func);

    // --------------- LossDeriv --------------- //
    typedef row_vector(*rvd_F_rvd_rvd)(const row_vector&, const row_vector&);

    row_vector SqLossDeriv(const row_vector& out, const row_vector& target);
    row_vector CrossEntropyLossDeriv(const row_vector& out, const row_vector& target);

    std::istream& operator>>(std::istream& str, rvd_F_rvd_rvd& func);
    std::ostream& operator<<(std::ostream& str, rvd_F_rvd_rvd& func);

    // --------------- Pool --------------- //
    typedef double(*d_F_mat)(const matrix&);

    double MaxPool(const matrix& mat);
    double AvgPool(const matrix& mat);

    std::istream& operator>>(std::istream& str, d_F_mat& func);
    std::ostream& operator<<(std::ostream& str, d_F_mat& func);

    // --------------- PoolDeriv --------------- //

    typedef matrix(*mat_F_mat_d)(const matrix&, double);

    matrix MaxPoolDeriv(const matrix& mat, double grad);
    matrix AvgPoolDeriv(const matrix& mat, double grad);

    std::istream& operator>>(std::istream& str, mat_F_mat_d& func);
    std::ostream& operator<<(std::ostream& str, mat_F_mat_d& func);

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