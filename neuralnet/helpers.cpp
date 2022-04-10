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
    matrix Sigmoid_Deriv(const row_vector& in) {
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
    matrix Tanh_Deriv(const row_vector& in) {
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
    matrix ReLU_Deriv(const row_vector& in) {
        matrix ret(in.size(), in.size());

        for (int i = 0; i < in.rows(); i++) {
            for (int j = 0; j < in.cols(); j++) ret(i, j) = (i == j) ? (in(i) > 0) : 0;
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

    matrix Softmax_Deriv(const row_vector& in) {
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

    row_vector Sq_Loss_Deriv(const row_vector& out, const row_vector& target) {
        if(out.size() != target.size()) throw std::runtime_error("Sq_Loss_Deriv : out and target sizes don't match\n");

        row_vector ret(out.size());
        for (int i = 0; i < out.size(); i++) ret(i) = out(i) - target(i);

        return ret;
    }

    double Sq_Loss(const row_vector& out, const row_vector& target)
    {
        if (out.size() != target.size()) throw std::runtime_error("Sq_Loss : out and target sizes don't match\n");

        double ret = 0;
        for (int i = 0; i < out.size(); i++) ret += (out(i) - target(i)) * (out(i) - target(i));

        return ret;
    }

    double Default_Random(int in, int out) {
        static std::random_device rd;
        std::mt19937 gen(rd());

        std::normal_distribution<double> normal(0, sqrt(1. / in));
        return normal(gen);
    }

    double Cross_Entropy_Loss(const row_vector& out, const row_vector& target) {
        if(out.size() != target.size()) throw std::runtime_error("Cross_Entropy_Loss : out and target sizes don't match\n");

        double ret = 0.;

        for (int i = 0; i < out.size(); i++) {
            if (target(i) == 1) ret = -log(out(i));
        }

        return ret;
    }

    row_vector Cross_Entropy_Loss_Deriv(const row_vector& out, const row_vector& target) {
        if (out.size() != target.size()) throw std::runtime_error("Cross_Entropy_Loss_Deriv : out and target sizes don't match\n");

        row_vector ret(out.size());

        for (int i = 0; i < out.size(); i++) {
            ret(i) = target(i) ? -1. / out(i) : 0;
        }

        return ret;
    }

    matrix RowVec2Matrix(const row_vector& rv, int x, int y) {
        if (rv.size() != x * y) throw std::runtime_error("RowVec2Matrix : vector size doesnt't match to matrix size\n");

        matrix ret(x, y);
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) ret(i, j) = rv(i * x + y);
        }

        return ret;
    }

    row_vector Matrix2RowVec(const matrix& mat) {
        row_vector ret(mat.rows() * mat.cols());

        for (int i = 0; i < mat.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) ret(i * mat.rows() + j) = mat(i, j);
        }

        return ret;
    }
}
