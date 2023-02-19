#pragma once
#include "Layer.h"

class LeakyReluLayer : public Layer
{
public:
	uint32_t inputMatrixSize;
	uint32_t outputMatrixSize;
	uint32_t weightMatrixSize;
	
	float* inputMatrix;
	float* productMatrix;
	float* activationMatrix;
	
	float* weightMatrix;
	float* biasMatrix;

	float* inputDerivativeMatrix;
	float* productDerivativeMatrix;
	float* activationDerivativeMatrix;

	uint32_t* weightDerivativeMatrixDisplacement;
	uint32_t* biasDerivativeMatrixDisplacement;
	float* dynamicDerivativeMatrix;
	/*float* weightDerivativeMatrix;
	float* biasDerivativeMatrix;*/
	
	LeakyReluLayer(uint32_t outputMatrixSize)
	{
		this->outputMatrixSize = outputMatrixSize;
	};
	
	~LeakyReluLayer() {};

	void AssignInputMatrixSize(uint32_t inputMatrixSize) override
	{
		this->inputMatrixSize = inputMatrixSize;
		weightMatrixSize = inputMatrixSize * outputMatrixSize;
	}

	uint32_t GetOutputMatrixSize() override
	{
		return outputMatrixSize;
	}

	void LoadLayerSpecs(
		std::vector<StaticMatrixInfo>& staticParams,
		std::vector<DynamicMatrixInfo>& dynamicParams
	) override
	{
		staticParams.emplace_back(StaticMatrixInfo{ outputMatrixSize, productMatrix });
		staticParams.emplace_back(StaticMatrixInfo{ outputMatrixSize, activationMatrix });
		staticParams.emplace_back(StaticMatrixInfo{ outputMatrixSize, productDerivativeMatrix });
		staticParams.emplace_back(StaticMatrixInfo{ inputMatrixSize, inputDerivativeMatrix });

		dynamicParams.emplace_back(DynamicMatrixInfo{ weightMatrixSize, weightMatrix, 0 });
		weightDerivativeMatrixDisplacement = &dynamicParams.back().displacement;
		dynamicParams.emplace_back(DynamicMatrixInfo{ outputMatrixSize, biasMatrix, 0 });
		biasDerivativeMatrixDisplacement = &dynamicParams.back().displacement;
	}

	float* GetOutputMatrix() override
	{
		return activationMatrix;
	}
	
	void AssignInputMatrix(float* inputMatrix) override
	{
		this->inputMatrix = inputMatrix;
	}
	
	float* GetInputDerivativeMatrix() override
	{
		return inputDerivativeMatrix;
	}

	void AssignOutputDerivativeMatrix(float* outputDerivativeMatrix) override
	{
		this->activationDerivativeMatrix = outputDerivativeMatrix;
	}

	void AssignDynamicDerivativeMatrix(float* dynamicDerivativeMatrix) override
	{
		this->dynamicDerivativeMatrix = dynamicDerivativeMatrix;
	}

	void Print() override
	{
		printf("LeakyReluLayer\n");
	}
};