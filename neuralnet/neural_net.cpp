#include "pch.h"
#include "neural_net.h"

namespace Nets
{
    Neural_Net::Neural_Net(std::vector<Layer*> layers_, double (*Error_Part_Deriv_)(double, double)) {
        layers = layers_;
        Error_Part_Deriv = Error_Part_Deriv_;

        Manage_In_Sizes(layers.front()->Input_Size());
    }
    Neural_Net::Neural_Net(const char* path, double (*Error_Part_Deriv_)(double, double)) {
        Error_Part_Deriv = Error_Part_Deriv_;
        Load(path);
    }
    Neural_Net::~Neural_Net() {
        for (auto e : layers) delete e;
    }

    Neural_Net::Neural_Net(const Neural_Net& org) {
        Error_Part_Deriv = org.Get_Error_Deriv();

        layers = org.Layers();
    }

    std::vector<Layer*> Neural_Net::Layers() const {
        std::vector<Layer*> ret;

        for (auto e : layers) ret.push_back(e->Clone());

        return ret;
    }
    Neural_Net::dfdd Neural_Net::Get_Error_Deriv() const { return Error_Part_Deriv; }

    void Neural_Net::Manage_In_Sizes(int input_size) {
        for (auto lay : layers) {
            lay->Set_In_Size(input_size);
            input_size = lay->Output_Size();
        }
    }
    void Neural_Net::Manage_Out_Sizes(int output_size) {
        for (int i = (int)layers.size() - 1; i >= 0; i--) {
            layers[i]->Set_Out_Size(output_size);
            output_size = layers[i]->Input_Size();
        }
    }

    void Neural_Net::Universal_Activation(double (*Act_Func)(double), double (*Act_Func_Deriv)(double)) {
        for (auto lay : layers) lay->Set_Functions(Act_Func, Act_Func_Deriv);
    }
    void Neural_Net::Universal_Lrate(double lrate) {
        for (auto lay : layers) lay->Set_Lrate(lrate);
    }
    void Neural_Net::Universal_Bias_Lrate(double bias_lrate) {
        for (auto lay : layers) lay->Set_Bias_Lrate(bias_lrate);
    }

    row_vector Neural_Net::Query(row_vector input) {
        for (auto lay : layers) input = lay->Forward(input);

        return input;
    }
    row_vector Neural_Net::Query(const std::vector<double>& input) {
        return Query(Vec2Eig<double>(input));
    }

    row_vector Neural_Net::Back_Query(row_vector grads) {
        for (int i = layers.size() - 1; i >= 0; i--) grads = layers[i]->Backward(grads);

        return grads;
    }
    row_vector Neural_Net::Back_Query(const std::vector<double>& grads) {
        return Back_Query(Vec2Eig(grads));
    }

    row_vector Neural_Net::Train(const row_vector& input, const row_vector& target) {
        row_vector grads = Query(input);

        for (int i = 0; i < grads.size(); i++) grads(i) = Error_Part_Deriv(grads(i), target(i));

        return Back_Query(grads);
    }
    row_vector Neural_Net::Train(const std::vector<double>& input, const std::vector<double>& target) {
        return Train(Vec2Eig(input), Vec2Eig(target));
    }

    void Neural_Net::Save(std::ostream& stream) {
        stream << layers.size() << '\n';

        for (auto lay : layers) lay->Write(stream);
    }
    void Neural_Net::Save(const std::string& path) {
        std::string ord = "type nul > ";
        ord += path;
        system(ord.c_str());

        std::ofstream file(path);
        Save(file);
    }
    void Neural_Net::Save(const char* path) {
        Save(std::string(path));
    }

    void Neural_Net::Load(std::istream& stream) {
        for (auto e : layers) delete e;
        layers.clear();

        int n;
        stream >> n;

        if (n < 0) throw std::runtime_error("bad data given to load");
        for (int i = 0; i < n; i++) {
            int ltype;
            stream >> ltype;

            switch (ltype) {
            case DENSE:
                layers.push_back(new DenseL);
                stream >> layers.back();

                break;
            case ACT:
                layers.push_back(new ActL);
                stream >> layers.back();

                break;
            default:
                throw std::runtime_error("bad data given to load");
                break;
            }
        }
    }
    void Neural_Net::Load(const std::string& path) {
        std::ifstream file(path);
        Load(file);
    }
    void Neural_Net::Load(const char* path) {
        std::ifstream file(path);
        Load(file);
    }
}

std::istream& operator>>(std::istream& istr, Nets::Neural_Net& net) {
    net.Load(istr);
    return istr;
}
std::ostream& operator<<(std::ostream& ostr, Nets::Neural_Net& net) {
    net.Save(ostr);
    return ostr;
}