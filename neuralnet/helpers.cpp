#include "pch.h"
#include "helpers.h"

namespace Nets {

    double Scale(double val, double mini1, double maxi1, double mini2, double maxi2) {
        return (val - mini1) / (maxi1 - mini1) * (maxi2 - mini2) + mini2;
    }

    row_vector Clip(const row_vector& rv, double mini, double maxi)
    {
        row_vector ret = rv;
        for (auto& e : ret) {
            e = std::min(e, maxi);
            e = std::max(e, mini);
        }

        return ret;
    }

    row_vector Sigmoid(const row_vector& in) {
        row_vector ret(in.size());

        for (int i = 0; i < in.size(); i++) ret(i) = 1 / (1 + exp(-in(i)));

        return ret;
    }
    matrix SigmoidDeriv(const row_vector& in) {
        row_vector sigmoid = Sigmoid(in);
        matrix ret(in.size(), in.size());

        for (int i = 0; i < ret.rows(); i++) {
            for (int j = 0; j < ret.cols(); j++) ret(i, j) = (i == j) ? sigmoid(i) * (1 - sigmoid(i)) : 0;
        }

        return ret;
    }

    row_vector Tanh(const row_vector& in) {
        row_vector ret = in;
        for (auto& e : ret) e = tanh(e);

        return ret;
    }
    matrix TanhDeriv(const row_vector& in) {
        row_vector tanh = Tanh(in);
        matrix ret(in.size(), in.size());

        for (int i = 0; i < ret.rows(); i++) {
            for (int j = 0; j < ret.cols(); j++) ret(i, j) = (i == j) ? 1 - tanh(i) * tanh(i) : 0;

        }

        return ret;
    }

    row_vector ReLU(const row_vector& in) {
        row_vector ret = in;

        for (auto& e : ret) e = std::max(0., e);

        return ret;
    }

    matrix ReLUDeriv(const row_vector& in) {
        matrix ret(in.size(), in.size());

        for (int i = 0; i < ret.rows(); i++) {
            for (int j = 0; j < ret.cols(); j++) ret(i, j) = (i == j) ? (in(i) > 0) : 0;
        }

        return ret;
    }

    row_vector Softmax(const row_vector& in){
        double sum = 0.;
        for (auto x : in) sum += exp(x);

        row_vector ret = in;
        for (auto& x : ret) x = exp(x) / sum;

        return ret;
    }

    matrix SoftmaxDeriv(const row_vector& in) {
        row_vector softmax = Softmax(in);
        
        matrix ret(in.size(), in.size());

        for (int i = 0; i < in.size(); i++) {
            for (int j = 0; j < in.size(); j++) {
                if (j == i) ret(i, j) = softmax(i) * (1 - softmax(i));
                else ret(i, j) = -softmax(i) * softmax(j);
            }
        }
        return ret;
    }

    row_vector SqLossDeriv(const row_vector& out, const row_vector& target) {
        if(out.size() != target.size()) throw std::runtime_error("Sq_Loss_Deriv : out and target sizes don't match\n");

        row_vector ret(out.size());
        for (int i = 0; i < out.size(); i++) ret(i) = out(i) - target(i);

        return ret;
    }

    double SqLoss(const row_vector& out, const row_vector& target)
    {
        if (out.size() != target.size()) throw std::runtime_error("Sq_Loss : out and target sizes don't match\n");

        double ret = 0;
        for (int i = 0; i < out.size(); i++) ret += (out(i) - target(i)) * (out(i) - target(i));

        return ret;
    }

    double DefaultRandom(int in, int out) {
        static std::random_device rd;
        std::mt19937 gen(rd());

        std::normal_distribution<double> normal(0, sqrt(1. / in));
        return normal(gen);
    }

    double CrossEntropyLoss(const row_vector& out, const row_vector& target) {
        if(out.size() != target.size()) throw std::runtime_error("Cross_Entropy_Loss : out and target sizes don't match\n");

        double ret = 0.;

        for (int i = 0; i < out.size(); i++) {
            if (target(i) == 1) ret = -log(out(i));
        }

        return ret;
    }

    row_vector CrossEntropyLossDeriv(const row_vector& out, const row_vector& target) {
        if (out.size() != target.size()) throw std::runtime_error("Cross_Entropy_Loss_Deriv : out and target sizes don't match\n");

        row_vector ret(out.size());

        for (int i = 0; i < out.size(); i++) {
            ret(i) = target(i) ? -1. / out(i) : 0;
        }

        return ret;
    }

    std::vector<matrix> RowVecTo3D(const row_vector& rv) {
        int x = rv(rv.size() - 3), y = rv(rv.size() - 2), z = rv(rv.size() - 1);

        std::vector<matrix> ret;
        for (int i = 0; i < z; i++) {
            ret.push_back(matrix{ x, y });
            for (int j = 0; j < x; j++) {
                for (int k = 0; k < y; k++) ret.back()(j, k) = rv(i * x * y + j * y + k);
            }
        }

        return ret;
    }

    row_vector ThreeDToRowVec(const std::vector<matrix>& threed) {
        int z = threed.size(), x = threed[0].rows(), y = threed[0].cols();

        row_vector ret{ z * x * y + 3 };

        for (int i = 0; i < z; i++) {
            for (int j = 0; j < x; j++) {
                for (int k = 0; k < y; k++) ret(i * x * y + j * y + k) = threed[i](j, k);
            }
        }

        ret(z * x * y) = x;
        ret(z * x * y + 1) = y;
        ret(z * x * y + 2) = z;

        return ret;
    }

    double MaxPool(const matrix& mat) {
        return mat.maxCoeff();
    }
    matrix MaxPoolDeriv(const matrix& mat, double grad) {
        matrix ret = matrix::Zero(mat.rows(), mat.cols());

        double maxi = mat.maxCoeff();

        for (int x = 0; x < mat.rows(); x++) {
            for (int y = 0; y < mat.cols(); y++) {
                if (mat(x, y) == maxi) ret(x, y) = grad;
            }
        }

        return ret;
    }

    double AvgPool(const matrix& mat) {
        return mat.sum() / (mat.rows() * mat.cols());
    }
    matrix AvgPoolDeriv(const matrix& mat, double grad) {
        matrix ret{ mat.rows(), mat.cols() };

        ret.fill(grad);

        return ret;
    }

    std::vector<rvd_F_rvd> ActDecode{ nullptr, Sigmoid, Tanh, ReLU, Softmax };
    std::map<rvd_F_rvd, int> ActEncode{
        {nullptr, 0},
        {Sigmoid, 1},
        {Tanh, 2},
        {ReLU, 3},
        {Softmax, 4}
    };
    std::vector<mat_F_rvd> ActDerivDecode{ nullptr, SigmoidDeriv, TanhDeriv, ReLUDeriv, SoftmaxDeriv };
    std::map<mat_F_rvd, int> ActDerivEncode{
        {nullptr, 0},
        {SigmoidDeriv, 1},
        {TanhDeriv, 2},
        {ReLUDeriv, 3},
        {SoftmaxDeriv, 4}
    };

    std::vector<d_F_rvd_rvd> LossDecode{ nullptr, SqLoss, CrossEntropyLoss };
    std::map<d_F_rvd_rvd, int> LossEncode{
        {nullptr, 0},
        {SqLoss, 1},
        {CrossEntropyLoss, 2}
    };
    std::vector<rvd_F_rvd_rvd> LossDerivDecode{ nullptr, SqLossDeriv, CrossEntropyLossDeriv };
    std::map<rvd_F_rvd_rvd, int> LossDerivEncode{
        {nullptr, 0},
        {SqLossDeriv, 1},
        {CrossEntropyLossDeriv, 2}
    };
    
    std::vector<d_F_mat> PoolDecode{ nullptr, MaxPool, AvgPool };
    std::map<d_F_mat, int> PoolEncode{
        {nullptr, 0},
        {MaxPool, 1},
        {AvgPool, 2}
    };
    std::vector<mat_F_mat_d> PoolDerivDecode{ nullptr, MaxPoolDeriv, AvgPoolDeriv };
    std::map<mat_F_mat_d, int> PoolDerivEncode{
        {nullptr, 0},
        {MaxPoolDeriv, 1},
        {AvgPoolDeriv, 2}
    };

    std::istream& operator>>(std::istream& str, rvd_F_rvd& func)
    {        
        int id;
        str >> id;

        func = ActDecode[id];

        return str;
    }
    std::ostream& operator<<(std::ostream& str, rvd_F_rvd& func)
    {
        str << ActEncode[func] << ' ';
        return str;
    }

    std::istream& operator>>(std::istream& str, mat_F_rvd& func)
    {
        int id;
        str >> id;

        func = ActDerivDecode[id];

        return str;
    }
    std::ostream& operator<<(std::ostream& str, mat_F_rvd& func)
    {
        str << ActDerivEncode[func] << ' ';
        return str;
    }

    std::istream& operator>>(std::istream& str, d_F_rvd_rvd& func)
    {
        int id;
        str >> id;

        func = LossDecode[id];

        return str;
    }
    std::ostream& operator<<(std::ostream& str, d_F_rvd_rvd& func)
    {
        str << LossEncode[func] << ' ';
        return str;
    }

    std::istream& operator>>(std::istream& str, rvd_F_rvd_rvd& func)
    {
        int id;
        str >> id;

        func = LossDerivDecode[id];

        return str;
    }
    std::ostream& operator<<(std::ostream& str, rvd_F_rvd_rvd& func)
    {
        str << LossDerivEncode[func] << ' ';
        return str;
    }

    std::istream& operator>>(std::istream& str, d_F_mat& func) {
        int id;
        str >> id;

        func = PoolDecode[id];

        return str;
    }
    std::ostream& operator<<(std::ostream& str, d_F_mat& func) {
        str << PoolEncode[func] << ' ';
        return str;
    }

    std::istream& operator>>(std::istream& str, mat_F_mat_d& func) {
        int id;
        str >> id;

        func = PoolDerivDecode[id];

        return str;
    }
    std::ostream& operator<<(std::ostream& str, mat_F_mat_d& func) {
        str << PoolDerivEncode[func] << ' ';
        return str;
    }
}
