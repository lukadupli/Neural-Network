#pragma once

#include "layers.h"
#include "helpers.h"

namespace Nets {
	const int FLAT = 5;

	class FlattenL : public LayerCRTP<FlattenL> {
	private:
		int flat_sz;

		std::vector<Eigen::RowVector3i> cache;
	public:
		FlattenL() = default;
		FlattenL(int flat_sz_);

		row_vector Forward(const row_vector& in, bool rec = 0) override;
		row_vector Backward(const row_vector& grads) override;

		std::istream& Read(std::istream& stream) override;
		std::ostream& Write(std::ostream& stream) override;
	};
}