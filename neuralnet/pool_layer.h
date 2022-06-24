#pragma once

#include "layers.h"
#include "helpers.h"

namespace Nets {
	const int POOL = 4;

	class PoolL : public LayerCRTP<PoolL> {
	private:
		int pool_sz = 2;

		d_F_mat PoolFunc;
		mat_F_mat_d PoolDeriv;

		std::vector<std::vector<matrix>>* cache = new std::vector<std::vector<matrix>>;
	public:
		PoolL() = default;

		PoolL(int pool_sz_, d_F_mat PoolFunc_, mat_F_mat_d PoolDeriv_);
		PoolL(const PoolL& org);

		~PoolL();

		int PoolSize() const;

		d_F_mat GetPoolFunc() const;
		mat_F_mat_d GetPoolDeriv() const;

		row_vector Forward(const row_vector& in, bool rec = 0) override;
		row_vector Backward(const row_vector& grads) override;

		std::istream& Read(std::istream& stream) override;
		std::ostream& Write(std::ostream& stream) override;
	};
}