#define EXCLUD

#ifndef EXCLUDE

#include <iostream>
#include <filesystem>
#include <random>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../neuralnet/neural_net.h"

#define f first
#define s second

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace Nets;
using namespace Nets::Cells;

const string LOCATION = R"(C:\Users\lukad\OneDrive\Dokumenti\VS_Projects\Neural_Network\neural_net_data\)";

vector<pair<string, bool>> train_data
{
	{"good", 1},
	{"bad", 0},
	{"happy", 1},
	{"sad", 0},
	{"not good", 0},
	{"not bad", 1},
	{"not happy", 0},
	{"not sad", 1},
};

int cnt = 0;
map<string, int> word_id;

vector<pair<vector<double>, bool>> raw_train;

Neural_Net rnn({
	new RecL(
		new Basic(5, 5, 2,
			Neural_Net({
				new DenseL(10, 10, 0.3, 0.6),
				new ActL(Tanh, Tanh_Deriv),
				new DenseL(10, 2, 0.6, 1.2),
				//new ActL(Sigmoid, Sigmoid_Deriv)
			})
		),
	END),
	new ActL(Softmax, Softmax_Deriv)
	},
	Cross_Entropy_Loss_Deriv,
	Cross_Entropy_Loss
);

string full_path(const char* path) {
	return LOCATION + path;
}

vector<string> split(const string& to_split, const string& spacing) {
	vector<string> ret;

	size_t prev = 0, now = 0;
	while (now != to_split.size()) {
		now = to_split.find(spacing, prev);

		if (now == string::npos) now = to_split.size();

		ret.push_back("");
		for (size_t i = prev; i < now; i++) ret.back().push_back(to_split[i]);

		prev = now + 1;
	}

	return ret;
}

template<typename T> std::ostream& operator<<(std::ostream& ostr, const vector<T>& vec) {
	for (auto& e : vec) ostr << e << ", ";
	return ostr;
}

int main() {
	for (auto &data : train_data) {
		vector<string> words = split(data.first, " ");
		for (string &str : words) {
			if (word_id.find(str) == word_id.end()) word_id[str] = cnt++;
		}
	}

	rnn.Manage_In_Sizes(word_id.size());
	rnn.Manage_Out_Sizes(2);

	for (auto &data : train_data) {
		vector<string> words = split(data.first, " ");
		vector<double> in;
		in.assign(words.size() * word_id.size(), 0);

		for (int i = 0; i < words.size(); i++) {
			in[i * word_id.size() + word_id[words[i]]] = 1;
		}

		raw_train.push_back({ in, data.second });
	}

	//for (auto& data : raw_train) cout << data.first << '\n' << data.second << '\n';

	//cout << Softmax(Vec2Eig(vector<double>{ 0.1, 0.2 })) << '\n';

	int epochs;
	cout << "No. of epochs: "; cin >> epochs;

	while (epochs--) {
		for (auto& data : raw_train) {
			vector<double> target = { 0, 0 };
			target[data.second] = 1;

			cout << rnn.Train(data.first, target) << '\n';
			cout << rnn.Query(data.first) << '\n' << target << "\n\n";
		}
	}


	return 0;
}

#endif
