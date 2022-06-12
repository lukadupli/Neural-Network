#include "pch.h"
#include "neural_net.h"
#include "layers.h"

namespace Nets
{
    
    Neural_Net::Neural_Net(std::vector<Layer*> layers_, rvd_F_rvd_rvd Loss_Deriv_, d_F_rvd_rvd Loss_Func_) {
        layers = layers_;
        Loss_Deriv = Loss_Deriv_;
        Loss_Func = Loss_Func_;

        Manage_In_Sizes(layers.front()->Input_Size());
    }
    Neural_Net::Neural_Net(const char* path, rvd_F_rvd_rvd Loss_Deriv_, d_F_rvd_rvd Loss_Func_) {
        Loss_Deriv = Loss_Deriv_;
        Loss_Func = Loss_Func_;

        Load(path);
    }
    Neural_Net::~Neural_Net() {
        for (auto e : layers) delete e;
    }

    Neural_Net::Neural_Net(const Neural_Net& org) {
        Loss_Deriv = org.Get_Loss_Deriv();
        Loss_Func = org.Get_Loss_Func();

        layers = org.Layers_Copy();
    }

    std::vector<Layer*> Neural_Net::Layers() { return layers; }
    std::vector<Layer*> Neural_Net::Layers_Copy() const {
        std::vector<Layer*> ret;

        for (auto e : layers) ret.push_back(e->Clone());

        return ret;
    }
    rvd_F_rvd_rvd Neural_Net::Get_Loss_Deriv() const { return Loss_Deriv; }
    d_F_rvd_rvd Neural_Net::Get_Loss_Func() const { return Loss_Func; }

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

    void Neural_Net::Universal_Activation(rvd_F_rvd Act_Func, mat_F_rvd Act_Deriv) {
        for (auto lay : layers) lay->Set_Functions(Act_Func, Act_Deriv);
    }
    void Neural_Net::Universal_Lrate(double lrate) {
        for (auto lay : layers) lay->Set_Lrate(lrate);
    }
    void Neural_Net::Universal_Bias_Lrate(double bias_lrate) {
        for (auto lay : layers) lay->Set_Bias_Lrate(bias_lrate);
    }

    row_vector Neural_Net::Query(row_vector input, bool rec) {
        for (auto lay : layers) input = lay->Forward(input, rec);

        return input;
    }
    row_vector Neural_Net::Query(const std::vector<double>& input, bool rec) {
        return Query(Vec2Eig<double>(input), rec);
    }

    row_vector Neural_Net::Back_Query(row_vector grads) {
        for (int i = layers.size() - 1; i >= 0; i--) grads = layers[i]->Backward(grads);

        return grads;
    }
    row_vector Neural_Net::Back_Query(const std::vector<double>& grads) {
        return Back_Query(Vec2Eig(grads));
    }

    double Neural_Net::Train(const row_vector& input, const row_vector& target, bool rec) {
        row_vector out = Query(input, rec);

        double ret = 0;

        if (target.size() != out.size()) throw std::runtime_error("Neural_Net : given target doesn't match output size\n");
        if (Loss_Func) ret = Loss_Func(out, target);

        Back_Query(Loss_Deriv(out, target));

        return ret;
    }
    double Neural_Net::Train(const std::vector<double>& input, const std::vector<double>& target, bool rec) {
        return Train(Vec2Eig(input), Vec2Eig(target), rec);
    }

    void Neural_Net::Save(std::ostream& stream) {
        stream << layers.size() << '\n' << Loss_Func << ' ' << Loss_Deriv << '\n';

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
        for (auto e : layers) if(e) delete e;
        layers.clear();

        int n;
        stream >> n >> Loss_Func >> Loss_Deriv;

        if (n < 0) throw std::runtime_error("bad data given to load");
        for (int i = 0; i < n; i++) {
            int ltype;
            stream >> ltype;

            switch (ltype) {
            case DENSE:
                layers.push_back(new DenseL);

                break;
            case ACT:
                layers.push_back(new ActL);

                break;
            case REC:
                layers.push_back(new RecL);

                break;
            default:
                throw std::runtime_error("bad data given to load");
                break;
            }

            layers.back()->Read(stream);
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