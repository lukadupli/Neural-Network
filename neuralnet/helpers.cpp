#include "pch.h"
#include "helpers.h"

namespace Nets {

    double Scale(double val, double mini1, double maxi1, double mini2, double maxi2) {
        return (val - mini1) / (maxi1 - mini1) * (maxi2 - mini2) + mini2;
    }

    row_vector Sigmoid(const row_vector& in) {
        row_vector ret(in.size());

        for (int i = 0; i < in.size(); i++) ret(i) = 1 / (1 + exp(-in(i)));

        return ret;
    }
    row_vector Sigmoid_Deriv(const row_vector& in) {
        row_vector ret = Sigmoid(in);

        for (int i = 0; i < in.size(); i++) ret(i) = ret(i) * (1 - ret(i));

        return ret;
    }

    row_vector Tanh(const row_vector& in) {
        row_vector ret = in;
        for (auto& e : ret) e = tanh(e);

        return ret;
    }
    row_vector Tanh_Deriv(const row_vector& in) {
        row_vector ret = Tanh(in);
        for (auto& e : ret) e = 1 - e * e;

        return ret;
    }

    row_vector ReLU(const row_vector& in) {
        row_vector ret = in;

        for (auto& e : ret) e = std::max(0., e);

        return ret;
    }
    row_vector ReLU_Deriv(const row_vector& in) {
        row_vector ret = in;

        for (auto& e : ret) e = (e > 0);

        return ret;
    }

    row_vector Softmax(const row_vector& in){
        double sum = 0.;
        for (auto x : in) sum += exp(x);

        row_vector ret = in;
        for (auto& x : ret) x = exp(x) / sum;

        return ret;
    }

    row_vector Softmax_Deriv(const row_vector& in) {
        row_vector softmax = Softmax(in);
        
        row_vector ret = row_vector::Zero(in.size());

        for (int i = 0; i < in.size(); i++) {
            for (int j = 0; j < in.size(); j++) {
                if (j == i) ret(i) += softmax(i) * (1 - softmax(i));
                else ret(i) -= softmax(i) * softmax(j);
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
            ret(i) = target(i) == 1 ? -1. / out(i) : 0;
        }

        return ret;
    }
}
