#include "pch.h"
#include "dense_layer.h"

namespace Nets
{
    void DenseL::Weight_Init(int input, int output, bool del) {
        if (del) delete weights, cache;
        weights = new Eigen::MatrixXd(input_sz, output_sz);
        cache = row_vector(input_sz);

        bias = 0;

        for (int i = 0; i < weights->rows(); i++) {
            for (int j = 0; j < weights->cols(); j++) {
                weights->coeffRef(i, j) = Init_Random(input, output);
            }
        }
    }

    DenseL::DenseL(int input_sz_, int output_sz_, double lrate_, double bias_lrate_, double (*Init_Random_)(int, int)) {
        input_sz = input_sz_;
        output_sz = output_sz_;
        lrate = lrate_;

        if (isnan(bias_lrate_)) bias_lrate = lrate;
        else bias_lrate = bias_lrate_;

        Init_Random = Init_Random_;

        Weight_Init(input_sz, output_sz, 0);
    }

    DenseL::DenseL(const DenseL& org) {
        input_sz = org.Input_Size();
        output_sz = org.Output_Size();

        lrate = org.Lrate();
        bias_lrate = org.Bias_Lrate();

        cache = row_vector(input_sz);

        Init_Random = org.Init_Func();

        bias = org.Bias();
        weights = new Eigen::MatrixXd(org.Weights());
    }

    DenseL::~DenseL() { delete weights; }

    int DenseL::Input_Size() const { return input_sz; };
    void DenseL::Set_In_Size(int in) { Set_Size({ in, SAME_SIZE }); }

    int DenseL::Output_Size() const { return output_sz; };
    void DenseL::Set_Out_Size(int out) { Set_Size({ SAME_SIZE, out }); }

    void DenseL::Set_Size(const std::vector<int>& sizes) {
        if (sizes[0] > 0) input_sz = sizes[0];
        if (sizes[1] > 0) output_sz = sizes[1];

        Weight_Init(input_sz, output_sz);
    }

    double DenseL::Lrate() const { return lrate; }
    void DenseL::Set_Lrate(double new_lrate) { lrate = new_lrate; }

    double DenseL::Bias_Lrate() const { return bias_lrate; }
    void DenseL::Set_Bias_Lrate(double new_bias_lrate) { bias_lrate = new_bias_lrate; }

    double DenseL::Bias() const { return bias; }
    Eigen::MatrixXd DenseL::Weights() const { return *weights; }

    row_vector DenseL::Forward(row_vector input) {
        if (input.size() != input_sz) throw std::runtime_error("Dense layer: rececived query list doesn't match specified size\n");

        cache = input;
        input *= (*weights);

        for (int i = 0; i < output_sz; i++) input.coeffRef(i) += bias;

        return input;
    }

    row_vector DenseL::Backward(row_vector gradients) {
        if (gradients.size() != output_sz) throw std::runtime_error("Dense layer: rececived gradient list doesn't match specified size\n");

        row_vector new_grads = gradients * weights->transpose();

        for (int i = 0; i < weights->rows(); i++) {
            for (int j = 0; j < weights->cols(); j++) {
                weights->coeffRef(i, j) -= lrate * cache(i) * gradients(j);
            }
        }

        double dL_db = 0.;
        for (int i = 0; i < gradients.size(); i++) dL_db += gradients.coeffRef(i);

        bias -= bias_lrate * dL_db;

        return new_grads;
    }

    std::istream& DenseL::Read(std::istream& stream) {
        stream >> input_sz >> output_sz >> lrate >> bias_lrate;

        weights = new Eigen::MatrixXd(input_sz, output_sz);
        for (int i = 0; i < weights->rows(); i++) {
            for (int j = 0; j < weights->cols(); j++) stream >> weights->coeffRef(i, j);
        }

        stream >> bias;

        return stream;
    }
    std::ostream& DenseL::Write(std::ostream& stream) {
        stream << DENSE << '\n' << input_sz << ' ' << output_sz << ' ' << lrate << ' ' << bias_lrate << '\n' << *weights << '\n' << bias << '\n';
        return stream;
    }
}
