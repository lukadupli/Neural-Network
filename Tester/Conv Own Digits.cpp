#define EXCLUDE

#ifndef EXCLUDE


#include <iostream>
#include <filesystem>
#include "../neuralnet/neural_net.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#define f first
#define s second

constexpr auto LOCATION = "C:/Users/lukad/Desktop/MNIST/";

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace Nets;

char loc[500];

double squared_err(double o, double t) {
    return (o - t);
}

double Random_Full(int in, int out) {
    static random_device rd;
    mt19937 gen(rd());

    normal_distribution<double> normal{ 0, sqrt(1. / in) };
    cout << sqrt(1 / in) << '\n' << sqrt(1. / in) << '\n';
    return normal(gen);
}

Neural_Net net({
    new ConvL(3, 8, 0.05),
    new PoolL(2, MaxPool, MaxPoolDeriv),
    new FlattenL(14 * 14 * 8),
    new DenseL(14 * 14 * 8, 10, 0.05, 0.1),
    new ActL(Softmax, SoftmaxDeriv) },

    CrossEntropyLossDeriv,
    CrossEntropyLoss
);

const double MIN_IN = 0, MAX_IN = 1;
const double ZERO_OUT = 0, ONE_OUT = 1;

bool load = 1, save = 0;

string FullPath(const string& path) {
    return string(LOCATION) + path;
}

const string load_modelname = "convnet_MNIST_9665.txt", save_modelname = "a.txt";

void PrintDigit(const matrix& img) {
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            if (img(i, j) > 0.5) cout << fixed << setprecision(3) << img(i, j) << ' ';
            else cout << "0     ";
        }
        cout << '\n';
    }
}

int main()
{
    if (load) {
        net.Load(FullPath(load_modelname));
    }

    cout << "Loading...\n";

    vector<pair<vector<matrix>, int>> data;
    for (int digit = 0; digit <= 9; digit++) {
        int cnt = 0;
        while (1) {
            memset(loc, 0, sizeof loc);

            sprintf_s(loc, FullPath("my_own_images/%d_%d.png").c_str(), digit, cnt);

            if (!filesystem::exists(loc)) break;

            Mat img = imread(loc, IMREAD_GRAYSCALE);

            resize(img, img, Size(28, 28), INTER_LINEAR);

            data.push_back({ vector<matrix>{matrix{28, 28}}, digit });

            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    int pixel = (int)img.at<uchar>(i, j);
                    data.back().first.front()(i, j) = (Scale(pixel, 0, 255, MIN_IN, MAX_IN));

                    double mid = (MIN_IN + MAX_IN) / 2;
                    data.back().first.front()(i, j) = 2 * mid - data.back().first.front()(i, j);
                }
            }

            cnt++;
        }
    }
    
    int epochs;
    cout << "No. of epochs: "; cin >> epochs;
    cout << "Training...\n";
    for (int _ = 0; _ < epochs; _++) {
        for (auto& e : data) {
            row_vector target = row_vector::Zero(10);
            target[e.second] = 1;

            cout << "Loss: " << net.Train(ThreeDToRowVec(e.first), target) << '\n';
        }
    }

    cout << "Testing...\n";

    int corr = 0;
    for (auto& e : data) {
        auto output = net.Query(ThreeDToRowVec(e.first));
        cout << output << '\n';

        double maxi = -1; int ind = -1;

        for (int i = 0; i < output.size(); i++) {
            if (output(i) > maxi) {
                maxi = output(i);
                ind = i;
            }
        }

        cout << "Network's guess: " << ind << "\nReal value is: " << e.second << '\n';
        if (ind == e.second) corr++;
    }

    cout << "Performance: " << (double)corr / data.size() * 100 << "%";

    if (save) net.Save(FullPath(save_modelname));
}

#endif
