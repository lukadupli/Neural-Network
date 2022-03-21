// Tester.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define EXCLUD

#ifndef EXCLUDE


#include <iostream>
#include <filesystem>
#include "../neuralnet/neural_net.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#define f first
#define s second

constexpr auto LOCATION = "C:/Users/lukad/OneDrive/Dokumenti/VS_Projects/Neural_Network/neural_net_data/";

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace Nets;

char loc[500];

vector <pair <char, vector <double>>> train_data;

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

Neural_Net netorg({ new DenseL(28 * 28, 200, 0),
               new ActL(),
               new DenseL(200, 8, 0),
               new ActL() }
);

const double MIN_IN = 0.01, MAX_IN = 0.99;
const double ZERO_OUT = 0.01, ONE_OUT = 0.99;

bool load = 0, save = 0;

string full_path(const char* path) {
    return string(LOCATION) + string(path);
}

int main()
{
    if (load) netorg.Load(full_path("saved/net1.txt"));

    netorg.Universal_Lrate(0.6);
    netorg.Universal_Bias_Lrate(1.2);
    netorg.Universal_Activation(Sigmoid, Sigmoid_Deriv);

    for (char c = 'a'; c <= 'z'; c++) {
        int cnt = 0;
        while (1) {
            memset(loc, 0, sizeof loc);

            sprintf_s(loc, full_path("samples/training/%d_%d.png").c_str(), int(c), cnt);

            if (!filesystem::exists(loc)) break;

            Mat img = imread(loc, IMREAD_GRAYSCALE);

            resize(img, img, Size(28, 28), INTER_LINEAR);

            train_data.push_back({ c, {} });

            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    int pixel = (int)img.at<uchar>(i, j);
                    train_data.back().s.push_back(Scale(pixel, 0, 255, MIN_IN, MAX_IN));
                }
            }

            cnt++;
        }
    }

    int epochs;
    cout << "No. of epochs: "; cin >> epochs;

    while (epochs--) {
        for (auto e : train_data) {
            vector <double> target;

            target.assign(8, ZERO_OUT);
            for (int i = 0; i < 8; i++) {
                if (e.f & (1 << i)) target[i] = ONE_OUT;
            }

            netorg.Train(e.s, target);
        }
    }

    int testcnt = 0, corr = 0;

    Neural_Net net = netorg;

    cout << "Testing...\n";
    for (char c = 'a'; c <= 'z'; c++) {
        int cnt = 0;
        while (1) {
            memset(loc, 0, sizeof loc);

            sprintf_s(loc, full_path("samples/test/%d_%d.png").c_str(), int(c), cnt);

            if (!filesystem::exists(loc)) break;

            Mat img = imread(loc, IMREAD_GRAYSCALE);

            resize(img, img, Size(28, 28), INTER_LINEAR);

            vector <double> input;

            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    int pixel = (int)img.at<uchar>(i, j);
                    input.push_back(Scale(pixel, 0, 255, MIN_IN, MAX_IN));
                }
            }

            row_vector output = net.Query(input);

            double maxi = -1;
            int ind = 0;

            for (int i = 0; i < 8; i++) {
                if (output(i) >= (ZERO_OUT + ONE_OUT) / 2.) ind |= (1 << i);
                cout << fixed << setprecision(5) << output(i) << ' ';
            }
            cout << '\n';

            if (ind == c) corr++;

            cout << "Correct output: " << c << "\nNetwork output: " << char(ind) << "\n\n";

            testcnt++;
            cnt++;
        }
    }

    cout << "Result: " << (double)corr / testcnt << '\n';

    if (save) net.Save(full_path("saved/net1.txt"));

    return 0;
}

#endif


// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
