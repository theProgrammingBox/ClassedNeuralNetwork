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

	uint32_t* weightDerivativeMatrixDisplacement;
	uint32_t* biasDerivativeMatrixDisplacement;
	
	LeakyReluLayer(uint32_t outputMatrixSize)
	{
		this->outputMatrixSize = outputMatrixSize;
	};
	
	~LeakyReluLayer() {};

	void AssignInputMatrixSize(uint32_t inputMatrixSize) override
	{
		this->inputMatrixSize = inputMatrixSize;
		weightMatrixSize = inputMatrixSize * outputMatrixSize;
		printf("LeakyReluLayer: inputMatrixSize = %d, outputMatrixSize = %d, weightMatrixSize = %d\n", inputMatrixSize, outputMatrixSize, weightMatrixSize);
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

		dynamicParams.emplace_back(DynamicMatrixInfo{ weightMatrixSize, &weightMatrix, 0 });
		weightDerivativeMatrixDisplacement = &dynamicParams.back().displacement;
		dynamicParams.emplace_back(DynamicMatrixInfo{ outputMatrixSize, &biasMatrix, 0 });
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

	void Print() override
	{
		printf("LeakyReluLayer\n");
		PrintMatrix(weightMatrix, inputMatrixSize, outputMatrixSize, "weightMatrix");
		PrintMatrix(biasMatrix, 1, outputMatrixSize, "biasMatrix");
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
		/*
		cpuSgemmStridedBatched(
			false, false,
			productMatrix.columns, productMatrix.rows, inputMatrix->columns,
			&GLOBAL::ONEF,
			weightMatrix.matrix, weightMatrix.columns, 0,
			inputMatrix->matrix, inputMatrix->columns, 0,
			&GLOBAL::ZEROF,
			productMatrix.matrix, productMatrix.columns, 0,
			1);
		cpuSaxpy(productMatrix.totalSize, &GLOBAL::ONEF, biasMatrix.matrix, 1, productMatrix.matrix, 1);
		cpuLeakyRelu(productMatrix.matrix, activationMatrix.matrix, activationMatrix.totalSize);
		*/

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

	void BackPropagate(float dt) override
	{
		/*
		cpuLeakyReluDerivative(productMatrix.matrix, activationDerivativeMatrix->matrix, productDerivativeMatrix.matrix, productDerivativeMatrix.totalSize);
		cpuSgemmStridedBatched(
			true, false,
			inputDerivativeMatrix.columns, inputDerivativeMatrix.rows, productDerivativeMatrix.columns,
			&GLOBAL::ONEF,
			weightMatrix.matrix, weightMatrix.columns, 0,
			productDerivativeMatrix.matrix, productDerivativeMatrix.columns, 0,
			&GLOBAL::ZEROF,
			inputDerivativeMatrix.matrix, inputDerivativeMatrix.columns, 0,
			1);
		cpuSgemmStridedBatched(
			false, true,
			weightDerivativeMatrix.columns, weightDerivativeMatrix.rows, inputMatrix->rows,
			&GLOBAL::ONEF,
			productDerivativeMatrix.matrix, productDerivativeMatrix.columns, 0,
			inputMatrix->matrix, inputMatrix->columns, 0,
			&GLOBAL::ONEF,
			weightDerivativeMatrix.matrix, weightDerivativeMatrix.columns, 0,
			1);
		cpuSaxpy(biasDerivativeMatrix.totalSize, &GLOBAL::ONEF, productDerivativeMatrix.matrix, 1, biasDerivativeMatrix.matrix, 1);
		*/
		float* weightDerivativeMatrix = *dynamicDerivativeMatrixPointer + *weightDerivativeMatrixDisplacement;
		float* biasDerivativeMatrix = *dynamicDerivativeMatrixPointer + *biasDerivativeMatrixDisplacement;

		cpuLeakyReluDerivative(productMatrix, activationDerivativeMatrix, productDerivativeMatrix, outputMatrixSize);
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
		cpuSaxpy(outputMatrixSize, &GLOBAL::ONEF, productDerivativeMatrix, 1, biasDerivativeMatrix, 1);
	}
};