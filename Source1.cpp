#include <iostream>

using namespace std;

int Square(int x) {
	return x * x;
}

unsigned char* fn;
int (*fptr)(int);

int main() {
	fn = reinterpret_cast<unsigned char*>(Square);

	for (int i = 0; 1; i++) {
		unsigned char byte = *(Square + i);
		for (int j = 7; j >= 0; j--) {
			if ((1 << j) & byte) cout << 1;
			else cout << 0;
		}

		cout << ' ';
		if (byte == 0b11000011) break;
	}

	fptr = reinterpret_cast<int(*)(int)>(fn);
	cout << fptr(2);
}