#include "pch.h"
#include "layers.h"

namespace Nets
{
    Layer::dfii Layer::Init_Func() const { return Init_Random; }

    void Layer::Clear_Cache(){
        cache->clear();
    }

    Cells::Cell* Layer::Cell() const { return nullptr; }
    void Layer::Set_Init_Func(dfii New_Init) { Init_Random = New_Init; }
}

std::istream& operator>>(std::istream& istr, Nets::Layer* lay) { return lay->Read(istr); }
std::ostream& operator<<(std::ostream& ostr, Nets::Layer* lay) { return lay->Write(ostr); }
