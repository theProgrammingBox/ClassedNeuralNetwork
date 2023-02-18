#pragma once
#include "Header.h"

class Layer
{
public:
	Layer() {};
	virtual ~Layer() {};

	virtual void Print() = 0;
};