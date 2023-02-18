#pragma once
#include "Header.h"

class Layer
{
public:
	Layer() {};
	virtual ~Layer() {};
	
	virtual void AssignInputMatrix(float* inputMatrix, std::vector<DynamicLayerSpec>* dynamicLayerSpecs) = 0;
	virtual float* GetOutputMatrix() = 0;
	virtual void AssignOutputDerivativeMatrix(float* outputDerivativeMatrix) = 0;
	virtual float* GetInputDerivativeMatrix() = 0;
	virtual void Print() = 0;
};