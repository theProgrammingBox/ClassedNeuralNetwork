#pragma once
#include "Layer.h"

class LeakyReluLayer : public Layer
{
public:
	uint32_t activationMatrixSize;
	float* activationMatrix;
	
	uint32_t inputMatrixSize;
	float* inputMatrix;

	uint32_t activationDerivativeMatrixSize;
	float* activationDerivativeMatrix;

	uint32_t inputDerivativeMatrixSize;
	float* inputDerivativeMatrix;
	
	LeakyReluLayer(uint32_t outputMatrixSize)
	{
		activationMatrixSize = outputMatrixSize;
		activationMatrix = new float[activationMatrixSize];
	};
	
	~LeakyReluLayer() {};

	void AssignInputMatrix(float* inputMatrix, std::vector<DynamicLayerSpec>* dynamicLayerSpecs) override
	{
		this->inputMatrix = inputMatrix;
	}
	
	float* GetOutputMatrix() override
	{
		return activationMatrix;
	}

	void AssignOutputDerivativeMatrix(float* outputDerivativeMatrix) override
	{
		activationDerivativeMatrix = outputDerivativeMatrix;
	}

	float* GetInputDerivativeMatrix() override
	{
		return inputDerivativeMatrix;
	}

	void Print() override
	{
		printf("LeakyReluLayer\n");
	}
};