#define EXCLUD
#ifndef EXCLUDE

#include <iostream>
#include <filesystem>

#include "../neuralnet/neural_net.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace Nets;
using namespace Eigen;

const string DATASET = R"(C:\Users\lukad\Desktop\math_symbols\)";
const string NN_DIR = R"(C:\Users\lukad\Desktop\math_symbols\nn_models\)";
const int TRESHOLD = 230;

std::vector<std::string> GetFilenames(const string& path)
{
    namespace stdfs = std::filesystem;

    if (!stdfs::exists(path)) {
        std::cerr << "GetFilenames( " << path << "): no such path exists!\n";
        throw std::runtime_error("");
    }

    std::vector<std::string> filenames;
    const stdfs::directory_iterator end{};

    for (stdfs::directory_iterator iter{ path }; iter != end; ++iter)
    {
        if (stdfs::is_regular_file(*iter))
            filenames.push_back(iter->path().string());
    }

    return filenames;
}
void Treshold(cv::Mat& img, const uchar treshold) {
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            auto& val = img.at<uchar>(i, j);
            val = (val < treshold) ? 0 : 255;
        }
    }
}
matrix Treshold(const matrix& mat, const double treshold) {
    matrix ret{mat.rows(), mat.cols()};

    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            ret(i, j) = (mat(i, j) < treshold) ? 0 : 1;
        }
    }

    return ret;
}
Nets::matrix cvMat2Netsmatrix(const cv::Mat& img) {
    Nets::matrix ret{ img.rows, img.cols };

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) ret(i, j) = img.at<uchar>(i, j);
    }

    return ret;
}

vector<string> decode{ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "mul", "div", "=", "(", ")" };
vector<vector<matrix>> LoadData() {
    vector<vector<matrix>> ret;

    //cv::namedWindow("slika");
    for (int id = 0; id < 17; id++) {
        auto fnames = GetFilenames(DATASET + decode[id]);

        ret.push_back(vector<matrix>{});
        for (int i = 0; i < fnames.size(); i++) {
            auto fname = fnames[i];
            cv::Mat img = cv::imread(fname, cv::IMREAD_GRAYSCALE);
            cv::resize(img, img, cv::Size(28, 28));

            ret.back().push_back(cvMat2Netsmatrix(img));
            for (int i = 0; i < ret.back().back().rows(); i++) {
                for (int j = 0; j < ret.back().back().cols(); j++) {
                    ret.back().back()(i, j) = Scale(ret.back().back()(i, j), 0, 255, 0, 1);
                }
            }
        }
    }

    return ret;
}

void Train(Neural_Net& net, const string& modelname, const vector<vector<matrix>>& data) {
    vector<int> ptr(data.size());
    vector<int> order;

    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[i].size(); j++) {
            order.push_back(i);
        }
    }

    int epochs;
    cout << "No. of epochs: "; cin >> epochs;

    int step, save_step;
    cout << "How often would you like to be informed of progress? (image number): "; cin >> step;
    cout << "How often would you like to save your NN? (image number): "; cin >> save_step;

    for (int ep = 1; ep <= epochs; ep++) {
        cout << "---------- Epoch " << ep << " ----------\n";
        for (auto& e : ptr) e = 0;

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        shuffle(order.begin(), order.end(), std::default_random_engine(seed));

        double loss = 0;
        for (int i = 0; i < order.size(); i++){
            row_vector tar = row_vector::Zero(data.size());
            tar[order[i]] = 1;

            matrix in = data[order[i]][ptr[order[i]]++];
            loss += net.Train(ThreeDToRowVec(vector<matrix>{in}), tar);

            if ((i + 1) % step == 0) {
                cout << "Passed " << i + 1 << " images, average loss is: " << loss / step << '\n';
                if (isnan(loss)) {
                    cout << "Loss is NaN, breaking the training process...\n";
                    return;
                }

                loss = 0;
            }

            if((i + 1) % save_step == 0) net.Save(NN_DIR + modelname);
        }

        net.Save(NN_DIR + modelname);
    }
}
void Test(Neural_Net& net, const vector<vector<matrix>> &data) {
    cout << "Testing...\n";

    int total_corr = 0, total_cnt = 0;
    for (int id = 0; id < data.size(); id++) {
        cout << "Symbol " << decode[id] << "...\n";

        int digit_corr = 0;
        for (auto& input : data[id]) {
            auto res = net.Query(ThreeDToRowVec(vector<matrix>{input}));
            cout << res << "\n";
            double maxi = -1e9;
            int pred = -1;

            for (int i = 0; i < res.size(); i++) {
                if (res(i) > maxi) {
                    maxi = res(i);
                    pred = i;
                }
            }

            if (pred == id) digit_corr++;
        }

        total_corr += digit_corr;
        total_cnt += data[id].size();

        cout << "   Local accuracy: " << digit_corr << "/" << data[id].size() << ", " << (double)digit_corr / data[id].size() * 100 << "%\n";
        cout << "   Total accuracy until now: " << total_corr << "/" << total_cnt << ", " << (double)total_corr / total_cnt * 100 << "%\n\n";
    }
}

Neural_Net net{ {
    new FlattenL(28 * 28),
    new DenseL(28 * 28, 100, 0.01, 0.02),
    new ActL(ReLU, ReLUDeriv),
    new DenseL(100, 17, 0.01, 0.02),
    new ActL(Softmax, SoftmaxDeriv)
},
    CrossEntropyLossDeriv,
    CrossEntropyLoss
};

int main()
{
    std::cout << "Loading data...\n";
    auto data = LoadData();
    //net.Load(NN_DIR + "model1.txt");
    Train(net, "model_test.txt", data);

    Test(net, data);
    

    //net.Load(NN_DIR + "model1.txt");

}

#endif