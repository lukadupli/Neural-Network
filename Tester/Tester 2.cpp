#define EXCLUDE

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

const string LOCATION = R"(C:\Users\lukad\OneDrive\Dokumenti\VS_Projects\Neural_Network\neural_net_data\mood_guess\)";

Neural_Net rnn({
	new RecL(
		new Basic(18, 2,
			Neural_Net({
				new DenseL(18, 10, 0.01, 0.02),
				new ActL(Tanh, Tanh_Deriv),
				new DenseL(10, 2, 0.01, 0.02)
			})
		),
	END),
	new ActL(Softmax, Softmax_Deriv)
	},
	Cross_Entropy_Loss_Deriv,
	Cross_Entropy_Loss
);

string full_path(const string& path) { return LOCATION + path; }
string full_path(const char* path) { return LOCATION + path; }

vector<pair<string, bool>> load(const string& filename) {
	string path = full_path(filename);

	ifstream istr(path);

	vector<pair<string, bool>> ret{};

	string sentence = ""; bool mood = false;
	while (istr) {
		string word;
		istr >> word;

		if (word == "0" || word == "1") {
			mood = word[0] - '0';

			sentence.pop_back();
			ret.push_back({ sentence, mood });
			sentence = "";
		}
		else {
			sentence += word;
			sentence.push_back(' ');
		}
	}

	return ret;
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

map<string, int> word_id;

vector<pair<vector<double>, bool>> numerize(const vector<pair<string, bool>>& data, map<string, int>& word_id) {
	vector<pair<vector<double>, bool>> ret;

	for (auto& d : data) {
		vector<string> words = split(d.first, " ");
		vector<double> in;
		in.assign(words.size() * word_id.size(), 0);

		for (int i = 0; i < words.size(); i++) {
			if (word_id.find(words[i]) == word_id.end()) throw std::runtime_error("numerize: word " + words[i] + " is not in the dictionary\n");
			
			in[i * word_id.size() + word_id[words[i]]] = 1;
		}

		ret.push_back({ in, d.second });
	}

	return ret;
}

template<typename T1, typename T2> std::ostream& operator<<(std::ostream& ostr, const pair<T1, T2>& par) {
	ostr << par.first << ' ' << par.second;
	return ostr;
}
template<typename T> std::ostream& operator<<(std::ostream& ostr, const vector<T>& vec) {
	for (auto& e : vec) ostr << e << ", ";
	return ostr;
}

int main() {
	auto train_data = load(R"(samples\train.txt)");
	auto test_data = load(R"(samples\test.txt)");

	int cnt = 0;
	for (auto &data : train_data) {
		vector<string> words = split(data.first, " ");
		for (string &str : words) {
			if (word_id.find(str) == word_id.end()) word_id[str] = cnt++;
		}
	}

	rnn.Manage_In_Sizes(word_id.size());
	rnn.Manage_Out_Sizes(2);

	auto raw_train = numerize(train_data, word_id);
	auto raw_test = numerize(test_data, word_id);

	/*
	rnn.Load(full_path(R"(saved\mood_guess.txt)"));
	rnn.Layers()[0]->Cell()->Gate().Universal_Activation(Tanh, Tanh_Deriv);
	rnn.Layers()[1]->Set_Functions(Softmax, Softmax_Deriv);
	*/

	int epochs;
	cout << "No. of epochs: "; cin >> epochs;

	for (int i = 1; i <= epochs; i++) {
		double loss = 0;
		for (auto& data : raw_train) {
			vector<double> target = { 0, 0 };
			target[data.second] = 1;

			loss = rnn.Train(data.first, target);
		}

		if (i % 10 == 0) cout << "Loss after " << i << " epochs: " << loss << '\n';
	}

	cout << "\n";
	int correct = 0;
	for (int i = 0; i < raw_test.size(); i++) {
		row_vector out = rnn.Query(raw_test[i].first);
		cout << out << '\n';

		string mood = out(0) > out(1) ? "bad" : "good", actual_mood = raw_test[i].second ? "good" : "bad";
		if (mood == actual_mood) correct++;
		
		cout << "You said: " << test_data[i].first << ".\nNetwork says your mood is " << mood << ".\nYour actual mood is " << actual_mood << ".\n\n";
	}

	cout << "Accuracy: " << (double)correct / raw_test.size() << '\n';
	//rnn.Save(full_path(R"(saved\mood_guess.txt)"));

	return 0;
}

#endif
