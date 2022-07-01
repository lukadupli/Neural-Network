#define EXCLUDE
#ifndef EXCLUDE

#include <iostream>
#include <filesystem>
#include <algorithm>
#include <random>
#include <chrono>

#include "../neuralnet/neural_net.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace Nets;

const std::string IMAGES_LOC = R"(C:\Users\lukad\Desktop\math_symbols\)";
const std::string NETS_LOC = R"(C:\Users\lukad\OneDrive\Dokumenti\MNIST_models\)";

void Treshold(cv::Mat& img, const uchar treshold) {
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            auto& val = img.at<uchar>(i, j);
            val = (val < treshold) ? 0 : 255;
        }
    }
}

Nets::matrix cvMat2Netsmatrix(const cv::Mat& img) {
    Nets::matrix ret{ img.rows, img.cols };

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) ret(i, j) = img.at<uchar>(i, j);
    }

    return ret;
}

void PrepareAsInput(matrix& input) {
    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) input(i, j) = Scale(input(i, j), 0, 255, 0, 1);
    }
}

std::vector<std::string> GetFilenames(const string& path)
{
    namespace stdfs = std::filesystem;

    std::vector<std::string> filenames;
    const stdfs::directory_iterator end{};

    for (stdfs::directory_iterator iter{ path }; iter != end; ++iter)
    {
        if (stdfs::is_regular_file(*iter))
            filenames.push_back(iter->path().string());
    }

    return filenames;
}

vector<string> decode{ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "mul", "div", "=", "(", ")" };
vector<vector<cv::Mat>> slicings;
vector<cv::Mat> blocks;

int main() {
    cout << "Loading data...\n";
    
    for (int i = 0; i < 4; i++) {
        blocks.push_back(cv::imread(IMAGES_LOC + "set" + to_string(i + 1) + ".jpg"));
    }

    for (int i = 0; i < decode.size(); i++) slicings.push_back(vector<cv::Mat>{});

    cout << "Slicing...\n";
    for (int block = 0; block < 3; block++) {
        for (int xslice = 0; xslice < 59 * 15; xslice += 59) {
            for (int yslice = 0; yslice < 59 * 37; yslice += 59) {
                slicings[xslice / 59].push_back(cv::Mat{ blocks[block], cv::Rect(yslice, xslice, 59, 59) });
            }
        }
    }

    //block 3:
    for (int xslice = 0; xslice < 59 * 6; xslice += 59) {
        bool bracket = (xslice / 59) % 2;
        for (int yslice = 0; yslice < 59 * 37; yslice += 59) {
            slicings[15 + bracket].push_back(cv::Mat{ blocks[3], cv::Rect(yslice, xslice, 59, 59) });
        }
    }

    cout << "Writing data...\n";
    for (int i = 0; i < decode.size(); i++) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        
        shuffle(slicings[i].begin(), slicings[i].end(), std::default_random_engine(seed));

        for (int j = 0; j < slicings[i].size(); j++) {
            string fname = IMAGES_LOC + decode[i] + "\\" + to_string(j) + ".jpg";
            cv::imwrite(fname, slicings[i][j]);
        }
    }

	return 0;
}

#endif