#include "pch.h"
#include "cells.h"

int Nets::Cells::Cell::Input_Size() const { return input_sz; }
int Nets::Cells::Cell::Hidden_Size() const { return hidden_sz; }

Nets::Neural_Net& Nets::Cells::Cell::Gate() const
{
	Neural_Net net;
	return net;
}

Nets::Neural_Net& Nets::Cells::Cell::Update_Gate() const
{
	Neural_Net net;
	return net;
}

Nets::Neural_Net& Nets::Cells::Cell::Reset_Gate() const
{
	Neural_Net net;
	return net;
}

Nets::Neural_Net& Nets::Cells::Cell::Output_Gate() const
{
	Neural_Net net;
	return net;
}

std::istream& operator>>(std::istream& str, Nets::Cells::Cell* c) { c->Read(str); return str; }
std::ostream& operator<<(std::ostream& str, Nets::Cells::Cell* c) { c->Write(str); return str; }