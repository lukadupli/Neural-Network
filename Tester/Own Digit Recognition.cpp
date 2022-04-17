
#define EXCLUD

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

Neural_Net net({ new DenseL(28 * 28, 200, 0),
               new ActL(),
               new DenseL(200, 8, 0),
               new ActL() }
);

const double MIN_IN = 0, MAX_IN = 1;
const double ZERO_OUT = 0, ONE_OUT = 1;

bool load = 1, save = 0;

string FullPath(const char* path) {
    return string(LOCATION) + string(path);
}

int main()
{
    if (load) net.Load(FullPath("netmodel1.txt"));

    net.Universal_Lrate(0.6);
    net.Universal_Bias_Lrate(1.2);
    net.Universal_Activation(Sigmoid, Sigmoid_Deriv);

    int corr = 0, cnt_all = 0;

    for (int digit = 0; digit <= 9; digit++) {
        int cnt = 0;
        while (1) {
            memset(loc, 0, sizeof loc);

            sprintf_s(loc, FullPath("my_own_images/%d_%d.png").c_str(), digit, cnt);

            if (!filesystem::exists(loc)) break;

            Mat img = imread(loc, IMREAD_GRAYSCALE);

            resize(img, img, Size(28, 28), INTER_LINEAR);

            vector<double> input;

            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    int pixel = (int)img.at<uchar>(i, j);
                    input.push_back(Scale(pixel, 0, 255, MIN_IN, MAX_IN));

                    double mid = (MIN_IN + MAX_IN) / 2;
                    input.back() = 2 * mid - input.back();
                }
            }

            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    if (input[i * 28 + j] > 0.5) cout << fixed << setprecision(3) << input[i * 28 + j] << ' ';
                    else cout << "0     ";
                }
                cout << '\n';
            }

            auto output = net.Query(input);
            cout << output << '\n';

            double maxi = -1; int ind = -1;

            for (int i = 0; i < output.size(); i++) {
                if (output(i) > maxi) {
                    maxi = output(i);
                    ind = i;
                }
            }

            cout << "Network's guess: " << ind << "\nReal value is: " << digit << '\n';
            if (ind == digit) corr++;

            cnt++;
        }

        cnt_all += cnt;
    }

    cout << "Performance: " << (double)corr / cnt_all * 100 << "%";
}

#endif
