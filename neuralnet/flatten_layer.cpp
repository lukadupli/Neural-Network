#include "pch.h"
#include "flatten_layer.h"

namespace Nets {
	FlattenL::FlattenL(int flat_sz_) { flat_sz = flat_sz_; }

	row_vector FlattenL::Forward(const row_vector& in, bool rec) {
		if (flat_sz < in.size() - 3) throw std::runtime_error("FlattenL : too large input received\n");

		if (!rec) cache.clear();
		cache.push_back(in.tail(3).cast<int>());

		row_vector ret = row_vector::Zero(flat_sz);
		ret.head(in.size() - 3) = in.head(in.size() - 3);
		
		return ret;
	}

	row_vector FlattenL::Backward(const row_vector& grads) {
		if (cache.empty()) throw std::runtime_error("Backward without previous forward\n");

		row_vector ret{ grads.size() + 3 };
		ret.head(grads.size()) = grads;
		ret.tail(3) = cache.back().cast<double>();

		cache.pop_back();

		return ret;
	}

	std::istream& FlattenL::Read(std::istream& str) {
		str >> flat_sz;
		return str;
	}
	std::ostream& FlattenL::Write(std::ostream& str) {
		str << FLAT << '\n' << flat_sz << '\n';
		return str;
	}
}
