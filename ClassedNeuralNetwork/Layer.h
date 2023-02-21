#pragma once
#include "Header.h"

class Layer
{
public:
	float** parameterDerivitiveMatrixPointer;	// Points to parameterDerivitiveMatrixLocation, used in conjunction with the parameter displacements in child classes

	Layer() {};
	virtual ~Layer() {};
	
	virtual void AssignInputMatrixSize(uint32_t inputMatrixSize) = 0;
	virtual uint32_t GetOutputMatrixSize() = 0;
	virtual void LoadLayerSpecs(std::vector<ComputationInfo>& staticParams, std::vector<ParameterInfo>& dynamicParams) = 0;
	virtual float* GetOutputMatrix() = 0;
	virtual void AssignInputMatrix(float* inputDerivativeMatrix) = 0;
	virtual float* GetInputDerivativeMatrix() = 0;
	virtual void AssignOutputDerivativeMatrix(float* outputDerivativeMatrix) = 0;
	virtual void ForwardPropagate() = 0;
	virtual void BackPropagate() = 0;
	virtual void Print() = 0;
	//virtual void Export(std::ofstream& file) = 0;
};