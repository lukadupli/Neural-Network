// Calculator.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <experimental/filesystem>

#include "../neuralnet/neural_net.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace Nets;

const string DATASET = R"(C:\Users\lukad\Desktop\calculator_dataset)";
const string NEURAL_NETS = R"(C:\Users\lukad\OneDrive\Dokumenti\MNIST_models)";

std::vector<std::string> GetFilenames(const string& path)
{
    namespace stdfs = std::experimental::filesystem;

    std::vector<std::string> filenames;
    const stdfs::directory_iterator end{};

    for (stdfs::directory_iterator iter{ path }; iter != end; ++iter)
    {
        if (stdfs::is_regular_file(*iter))
            filenames.push_back(iter->path().string());
    }

    return filenames;
}

Neural_Net net;

int main()
{
    net.Load(NEURAL_NETS + "convnet_MNIST_9722.txt");
    cout << net;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
