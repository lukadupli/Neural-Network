#include "pch.h"
#include "pool_layer.h"

namespace Nets{

	PoolL::PoolL(int pool_sz_, d_F_mat PoolFunc_, mat_F_mat_d PoolDeriv_) {
		pool_sz = pool_sz_;
		PoolFunc = PoolFunc_;
		PoolDeriv = PoolDeriv_;
	}

	PoolL::PoolL(const PoolL& org) {
		pool_sz = org.PoolSize();

		PoolFunc = org.GetPoolFunc();
		PoolDeriv = org.GetPoolDeriv();
	}

	PoolL::~PoolL() { delete cache; }

	int PoolL::PoolSize() const { return pool_sz; }

	d_F_mat PoolL::GetPoolFunc() const { return PoolFunc; }
	mat_F_mat_d PoolL::GetPoolDeriv() const { return PoolDeriv; }

	row_vector PoolL::Forward(const row_vector& in, bool rec) {
		std::vector<matrix> real_in = RowVecTo3D(in), out;

		if (!rec) cache->clear();
		cache->push_back(real_in);

		int newrows = real_in[0].rows() / pool_sz + bool(real_in[0].rows() % pool_sz);
		int newcols = real_in[0].cols() / pool_sz + bool(real_in[0].cols() % pool_sz);

		for (auto& mat : real_in) {
			out.push_back(matrix{ newrows, newcols });

			for (int x = 0; x < newrows; x++) {
				for (int y = 0; y < newcols; y++)
					out.back()(x, y) = 
					PoolFunc(mat.block(x * pool_sz, y * pool_sz, 
						std::min((long long)pool_sz, mat.rows() - x * pool_sz), 
						std::min((long long)pool_sz, mat.cols() - y * pool_sz)));
					
			}
		}

		return ThreeDToRowVec(out);
	}

	row_vector PoolL::Backward(const row_vector& grads) {
		if (cache->empty()) throw std::runtime_error("Backward without previous forward\n");

		std::vector<matrix> real_grads = RowVecTo3D(grads), ret;

		for (int z = 0; z < real_grads.size(); z++) {
			int r = cache->back()[z].rows(), c = cache->back()[z].cols();

			ret.push_back(matrix{ r, c });

			for (int x = 0; x < real_grads[z].rows(); x++) {
				for (int y = 0; y < real_grads[z].cols(); y++) {
					ret.back().block(x * pool_sz, y * pool_sz, std::min(pool_sz, r - x * pool_sz), std::min(pool_sz, c - y * pool_sz)) =
					PoolDeriv(cache->back()[z].block(x * pool_sz, y * pool_sz, std::min(pool_sz, r - x * pool_sz), std::min(pool_sz, c - y * pool_sz)), real_grads[z](x, y));
				}
			}
		}

		cache->pop_back();
		
		return ThreeDToRowVec(ret);
	}

	std::istream& PoolL::Read(std::istream& str) {
		str >> PoolFunc >> PoolDeriv;

		return str;
	}

	std::ostream& PoolL::Write(std::ostream& str) {
		str << POOL << '\n';

		str << PoolFunc << ' ' << PoolDeriv << '\n';
		return str;
	}
}
