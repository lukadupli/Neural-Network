#define EXCLUD
#ifndef EXCLUDE

#include <iostream>

#include <Eigen/Dense>

#include "../neuralnet/neural_net.h"

using namespace std;
using namespace Nets;

typedef unsigned char ubyte;
typedef char byte;

const string LOCATION = R"(C:\Users\lukad\Desktop\MNIST\)";
string FullPath(const string& str) {
	return LOCATION + str;
}

std::ostream& operator<<(std::ostream& os, const ubyte& b) {
	for (int i = 7; i >= 0; i--) os << bool(b & (1 << i));

	return os;
}
std::ostream& bitprint(std::ostream& os, const int& n) {
	for (int i = 31; i >= 0; i--) os << bool(n & (1 << i));

	return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
	os << "[";
	for (int i = 0; i < vec.size() - 1; i++) os << vec[i] << ", ";
	if (!vec.empty()) os << vec.back();
	os << "]";
	return os;
}

int Read4ByteInt(std::istream& stream) {
	unsigned char buf[4];
	stream.read((char*)buf, 4);

	int ret = 0;
	for (int i = 0; i < 4; i++) {
		ret <<= 8;
		ret |= buf[i];
	}

	return ret;
}

std::vector<ubyte> ReadUbyteIdx1File(std::istream& stream) {
	for (int i = 0; i < 4; i++) stream.get();

	int size = Read4ByteInt(stream);

	std::vector<ubyte> ret;
	ret.resize(size);
	stream.read((char*)(&ret[0]), size);

	return ret;
}

std::vector<matrix>
ReadUbyteIdx3File(std::istream& stream) {
	for (int i = 0; i < 4; i++) stream.get();

	unsigned int size = Read4ByteInt(stream);
	unsigned int rows = Read4ByteInt(stream);
	unsigned int cols = Read4ByteInt(stream);

	std::vector<ubyte> bytes{ std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>() };

	std::vector<matrix> ret;
	for (int n = 0; n < size; n++) {
		ret.push_back(matrix(rows, cols));
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) ret.back()(i, j) = Scale((double)bytes[n * rows * cols + i * cols + j], 0, 255, 0, 1);
		}
	}

	return ret;
}

std::vector<std::vector<ubyte>>
ReadAndFlattenUbyteIdx3File(std::istream& stream) {
	for (int i = 0; i < 4; i++) stream.get();

	unsigned int size = Read4ByteInt(stream);
	unsigned int rows = Read4ByteInt(stream);
	unsigned int cols = Read4ByteInt(stream);

	std::vector<ubyte> bytes{ std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>() };

	std::vector<std::vector<ubyte>> ret(size);
	for (int n = 0; n < size; n++) {
		ret[n].resize(rows * cols);
		for (int i = 0; i < rows * cols; i++) {
			ret[n][i] = bytes[n * rows * cols + i];
		}
	}

	return ret;
}

Neural_Net net{ {
	new ConvL(3, 8, 0.001),
	new PoolL(2, MaxPool, MaxPoolDeriv),
	new ConvL(3, 8, 0.001),
	new PoolL(2, MaxPool, MaxPoolDeriv),
	new FlattenL(7 * 7 * 8 * 8),
	new DenseL(7 * 7 * 8 * 8, 10, 0.01, 0.02),
	new ActL(Softmax, SoftmaxDeriv)
	},
	CrossEntropyLossDeriv,
	CrossEntropyLoss
};

const string modelname = "convnet_MNIST_model3.txt";

void Train(bool load) {
	cout << "Loading data...\n";
	if (load) net.Load(FullPath(modelname));

	ifstream labfile(FullPath("train-labels.idx1-ubyte"), std::ios::binary);
	auto labels = ReadUbyteIdx1File(labfile);

	ifstream imfile(FullPath("train-images.idx3-ubyte"), std::ios::binary);
	auto images = ReadUbyteIdx3File(imfile);

	int how_many, step;
	cout << "How many images to take: "; cin >> how_many;
	cout << "How often would you like to be informed of progress (image number): "; cin >> step;
	cout << "Training...\n";

	if (how_many > images.size()) {
		cout << "Too much, " << images.size() << " is the maximum. Taking the whole dataset...\n";
		how_many = images.size();
	}

	double loss = 0;
	for (int i = 0; i < how_many; i++) {
		row_vector tar = row_vector::Zero(10);
		tar[labels[i]] = 1;

		loss += net.Train(ThreeDToRowVec(vector<matrix>{images[i]}), tar);
		if ((i + 1) % step == 0) {
			cout << "Passed " << i + 1 << " images, average loss is: " << loss / step << '\n';
			loss = 0;
			net.Save(FullPath(modelname));
		}
	}

	net.Save(FullPath(modelname));
}

void Test(bool load) {
	cout << "Loading data...\n";
	if (load) {
		net.Load(FullPath(modelname));
	}

	ifstream labfile(FullPath("test-labels.idx1-ubyte"), std::ios::binary);
	auto labels = ReadUbyteIdx1File(labfile);

	ifstream imfile(FullPath("test-images.idx3-ubyte"), std::ios::binary);
	auto images = ReadUbyteIdx3File(imfile);

	int correct = 0;
	cout << "Testing...\n";
	for (int i = 0; i < images.size(); i++) {
		

		auto out = net.Query(ThreeDToRowVec(vector<matrix>{images[i]}));
		bool ok = 1;

		double maxi = -1; int ind = -1;
		for (int j = 0; j < 10; j++) {
			if (out(j) > maxi) {
				ind = j;
				maxi = out(j);
			}
		}

		if (ind == labels[i]) correct++;

		if ((i + 1) % 1000 == 0) {
			cout << "Passed " << i + 1 << " images, accuracy is: " << (double)correct / (i + 1) << '\n';
		}
	}
}

int main() {
	Train(0);
	Test(0);

	return 0;
}

#endif