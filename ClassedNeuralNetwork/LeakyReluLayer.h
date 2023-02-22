#pragma once
#include "Layer.h"

class LeakyReluLayer : public Layer
{
public:
	uint32_t inputMatrixSize;	// Defined in network initialization during AssignInputMatrixSize
	uint32_t outputMatrixSize;	// Defined in constructor by user
	uint32_t weightMatrixSize;	// Defined in network initialization during AssignInputMatrixSize
	
	float* inputMatrix;					// Defined in network initialization during AssignInputMatrix
	float* weightMatrix;				// Defined in network initialization with the help of ParameterInfo and parameterMatrix
	float* productMatrix;				// Defined in network initialization with the help of ComputationInfo and computationMatrix
	float* biasMatrix;					// Defined in network initialization with the help of ParameterInfo and parameterMatrix
	float* activationMatrix;			// Defined in network initialization with the help of ComputationInfo and computationMatrix
	float* activationDerivativeMatrix;	// Defined in network initialization during AssignOutputDerivativeMatrix
	float* productDerivativeMatrix;		// Defined in network initialization with the help of ComputationInfo and computationMatrix
	float* biasDerivativeMatrix;		// Defined in network initialization with the help of ParameterInfo and parameterMatrix
	float* weightDerivativeMatrix;		// Defined in network initialization with the help of ParameterInfo and parameterMatrix
	float* inputDerivativeMatrix;		// Defined in network initialization with the help of ComputationInfo and computationMatrix
	
	
	LeakyReluLayer(uint32_t outputMatrixSize)
	{
		// sets the output matrix size defined by the user
		this->outputMatrixSize = outputMatrixSize;
	};
	
	~LeakyReluLayer() {};

	void AssignInputMatrixSize(uint32_t inputMatrixSize) override
	{
		// updates the input matrix size and weight matrix size because now we know all the dimensions of the layer
		this->inputMatrixSize = inputMatrixSize;
		weightMatrixSize = inputMatrixSize * outputMatrixSize;
	}

	uint32_t GetOutputMatrixSize() override
	{
		return outputMatrixSize;
	}

	void LoadLayerSpecs(
		std::vector<PartitionData>& computationPartitionData,
		std::vector<PartitionData>& parameterPartitionData,
		std::vector<PartitionData>& parameterDerivitivePartitionData
	) override
	{
		// adds all parameters related to this layer that is not the input matrix or output derivative matrix because they are defined by other layers or the network
		computationPartitionData.emplace_back(PartitionData{ outputMatrixSize, &productMatrix });
		computationPartitionData.emplace_back(PartitionData{ outputMatrixSize, &activationMatrix });
		computationPartitionData.emplace_back(PartitionData{ outputMatrixSize, &productDerivativeMatrix });
		computationPartitionData.emplace_back(PartitionData{ inputMatrixSize, &inputDerivativeMatrix });
		
		parameterPartitionData.emplace_back(PartitionData{ weightMatrixSize, &weightMatrix });
		parameterPartitionData.emplace_back(PartitionData{ outputMatrixSize, &biasMatrix });
		
		parameterDerivitivePartitionData.emplace_back(PartitionData{ weightMatrixSize, &weightDerivativeMatrix });
		parameterDerivitivePartitionData.emplace_back(PartitionData{ outputMatrixSize, &biasDerivativeMatrix });

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
		// reflection of relu
		for (size_t counter = size; counter--;)
			output[counter] = (((*(int32_t*)(input + counter) & 0x80000000) >> 31) * 0.9f + 0.1f) * input[counter];
	}

	void cpuLeakyReluDerivative(float* input, float* gradient, float* output, uint32_t size)
	{
		// reflection of relu derivative
		for (size_t counter = size; counter--;)
			output[counter] = (((*(int32_t*)(input + counter) & 0x80000000) >> 31) * 0.9f + 0.1f) * gradient[counter];
	}

	void ForwardPropagate() override
	{
		// multiplies the input matrix by the weight matrix and adds the bias matrix
		// finally applies the leaky relu activation function
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
		// first defines the weight derivative matrix and bias derivative matrix based on the parameterDerivitiveMatrixPointer and the displacement
		// then use output derivative matrix to calculate the product derivative matrix and bias derivative matrix
		// finally use the product derivative matrix to calculate the input derivative and weight derivative
		// parameter gradients are added instead of replaces due to batch gradient descent
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
		cpuSaxpy(outputMatrixSize, &GLOBAL::ONEF, productDerivativeMatrix, 1, biasDerivativeMatrix, 1);
		cpuSgemmStridedBatched(
			false, true,
			outputMatrixSize, inputMatrixSize, 1,
			&GLOBAL::ONEF,
			productDerivativeMatrix, outputMatrixSize, 0,
			inputMatrix, inputMatrixSize, 0,
			&GLOBAL::ONEF,
			weightDerivativeMatrix, outputMatrixSize, 0,
			1);
	}

	void Print() override
	{
		printf("LeakyReluLayer\n");
		printf("\n");
		PrintMatrix(inputMatrix, 1, inputMatrixSize, "inputMatrix");
		PrintMatrix(weightMatrix, inputMatrixSize, outputMatrixSize, "weightMatrix");
		PrintMatrix(biasMatrix, 1, outputMatrixSize, "biasMatrix");
		PrintMatrix(productMatrix, 1, outputMatrixSize, "productMatrix");
		PrintMatrix(activationMatrix, 1, outputMatrixSize, "activationMatrix");
		PrintMatrix(inputDerivativeMatrix, 1, inputMatrixSize, "inputDerivativeMatrix");
		PrintMatrix(biasDerivativeMatrix, 1, outputMatrixSize, "biasDerivativeMatrix");
		PrintMatrix(weightDerivativeMatrix, inputMatrixSize, outputMatrixSize, "weightDerivativeMatrix");
		PrintMatrix(productDerivativeMatrix, 1, outputMatrixSize, "productDerivativeMatrix");
		PrintMatrix(activationDerivativeMatrix, 1, outputMatrixSize, "activationDerivativeMatrix");
		printf("\n");
	}

	/*void Export(std::ofstream& file) override
	{
		file.write((char*)&inputMatrixSize, sizeof(uint32_t));
	}*/
};