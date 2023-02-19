#pragma once
#include "Header.h"

class Layer
{
public:
	Layer() {};
	virtual ~Layer() {};
	
	virtual void AssignInputMatrixSize(uint32_t inputMatrixSize) = 0;
	virtual uint32_t GetOutputMatrixSize() = 0;
	virtual void LoadLayerSpecs(
		std::vector<StaticMatrixInfo>& staticParams,
		std::vector<DynamicMatrixInfo>& dynamicParams
	) = 0;
	
	virtual float* GetOutputMatrix() = 0;
	virtual void AssignInputMatrix(float* inputDerivativeMatrix) = 0;
	virtual float* GetInputDerivativeMatrix() = 0;
	virtual void AssignOutputDerivativeMatrix(float* outputDerivativeMatrix) = 0;

	virtual void ForwardPropagate() = 0;
	virtual void BackPropagate(float dt) = 0;
	
	virtual void Print() = 0;

	float** dynamicDerivativeMatrixPointer;
};