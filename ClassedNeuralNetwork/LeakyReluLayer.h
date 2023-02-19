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

	float* activationDerivativeMatrix;
	float* productDerivativeMatrix;
	float* inputDerivativeMatrix;

	uint32_t weightDerivativeMatrixDisplacement;
	uint32_t biasDerivativeMatrixDisplacement;
	
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
		staticParams.emplace_back(StaticMatrixInfo{ outputMatrixSize, &productMatrix });
		staticParams.emplace_back(StaticMatrixInfo{ outputMatrixSize, &activationMatrix });
		staticParams.emplace_back(StaticMatrixInfo{ outputMatrixSize, &productDerivativeMatrix });
		staticParams.emplace_back(StaticMatrixInfo{ inputMatrixSize, &inputDerivativeMatrix });

		dynamicParams.emplace_back(DynamicMatrixInfo{ weightMatrixSize, &weightMatrix, &weightDerivativeMatrixDisplacement });
		dynamicParams.emplace_back(DynamicMatrixInfo{ outputMatrixSize, &biasMatrix, &biasDerivativeMatrixDisplacement });
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
	
	void cpuLeakyRelu(float* input, float* output, uint32_t size)
	{
		for (size_t counter = size; counter--;)
			output[counter] = (((*(int32_t*)(input + counter) & 0x80000000) >> 31) * 0.9f + 0.1f) * input[counter];
	}

	void cpuLeakyReluDerivative(float* input, float* gradient, float* output, uint32_t size)
	{
		for (size_t counter = size; counter--;)
			output[counter] = (((*(int32_t*)(input + counter) & 0x80000000) >> 31) * 0.9f + 0.1f) * gradient[counter];
	}

	void ForwardPropagate() override
	{
		cpuSgemmStridedBatched(
			false, false,
			outputMatrixSize, 1, inputMatrixSize,
			&GLOBAL::ONEF,
			weightMatrix, outputMatrixSize, 0,
			inputMatrix, inputMatrixSize, 0,
			&GLOBAL::ZEROF,
			productMatrix, outputMatrixSize, 0,
			1);
		cpuSaxpy(outputMatrixSize, &GLOBAL::ONEF, biasMatrix, 1, productMatrix, 1);
		cpuLeakyRelu(productMatrix, activationMatrix, outputMatrixSize);
	}

	void BackPropagate() override
	{
		float* weightDerivativeMatrix = *dynamicDerivativeMatrixPointer + weightDerivativeMatrixDisplacement;
		float* biasDerivativeMatrix = *dynamicDerivativeMatrixPointer + biasDerivativeMatrixDisplacement;

		printf("------------------------------------\n");
		PrintMatrix(productDerivativeMatrix, 1, outputMatrixSize, "productDerivativeMatrix");
		PrintMatrix(activationDerivativeMatrix, 1, outputMatrixSize, "activationDerivativeMatrix");
		cpuLeakyReluDerivative(productMatrix, activationDerivativeMatrix, productDerivativeMatrix, outputMatrixSize);
		PrintMatrix(productDerivativeMatrix, 1, outputMatrixSize, "productDerivativeMatrix");
		cpuSgemmStridedBatched(
			true, false,
			inputMatrixSize, 1, outputMatrixSize,
			&GLOBAL::ONEF,
			weightMatrix, outputMatrixSize, 0,
			productDerivativeMatrix, outputMatrixSize, 0,
			&GLOBAL::ZEROF,
			inputDerivativeMatrix, inputMatrixSize, 0,
			1);
		cpuSgemmStridedBatched(
			false, true,
			outputMatrixSize, inputMatrixSize, 1,
			&GLOBAL::ONEF,
			productDerivativeMatrix, outputMatrixSize, 0,
			inputMatrix, inputMatrixSize, 0,
			&GLOBAL::ONEF,
			weightDerivativeMatrix, outputMatrixSize, 0,
			1);
		PrintMatrix(weightDerivativeMatrix, inputMatrixSize, outputMatrixSize, "weightDerivativeMatrix");
		printf("------------------------------------\n");
		cpuSaxpy(outputMatrixSize, &GLOBAL::ONEF, productDerivativeMatrix, 1, biasDerivativeMatrix, 1);
	}

	void Print() override
	{
		printf("LeakyReluLayer\n\n");
		PrintMatrix(inputMatrix, 1, inputMatrixSize, "inputMatrix");
		PrintMatrix(weightMatrix, inputMatrixSize, outputMatrixSize, "weightMatrix");
		PrintMatrix(biasMatrix, 1, outputMatrixSize, "biasMatrix");
		PrintMatrix(productMatrix, 1, outputMatrixSize, "productMatrix");
		PrintMatrix(activationMatrix, 1, outputMatrixSize, "activationMatrix");
		
		float* weightDerivativeMatrix = *dynamicDerivativeMatrixPointer + weightDerivativeMatrixDisplacement;
		float* biasDerivativeMatrix = *dynamicDerivativeMatrixPointer + biasDerivativeMatrixDisplacement;
		
		PrintMatrix(inputDerivativeMatrix, 1, inputMatrixSize, "inputDerivativeMatrix");
		PrintMatrix(biasDerivativeMatrix, 1, outputMatrixSize, "biasDerivativeMatrix");
		PrintMatrix(weightDerivativeMatrix, inputMatrixSize, outputMatrixSize, "weightDerivativeMatrix");
		PrintMatrix(productDerivativeMatrix, 1, outputMatrixSize, "productDerivativeMatrix");
		PrintMatrix(activationDerivativeMatrix, 1, outputMatrixSize, "activationDerivativeMatrix");
		printf("\n");
	}
};