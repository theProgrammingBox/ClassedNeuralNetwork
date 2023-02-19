#pragma once
#include "Layer.h"

class NeuralNetwork
{
public:
	uint32_t inputMatrixSize;						// Defined in constructor by user
	float* inputMatrix;								// Created in constructor
	float* outputMatrix;							// Created later in initialization once the mass matrix is created
	float* outputDerivativeMatrix;					// Created earlier in initialization once last layer is created
	float* inputDerivativeMatrix;					// Created later in initialization once the mass matrix is created
	
	uint32_t computationMatrixSize;					// Size of the mass matrix for things that are not parameters
	uint32_t parameterMatrixSize;					// Size of the mass matrix for parameters
	uint32_t parameterDerivitiveMatrixSize;			// Size of the mass matrix for parameter derivitives
	uint32_t parameterDerivitiveMatrixDisplacement;	// Displacement of the mass parameter matrix, for Runge Kutta 4th order
	float* computationMatrix;						// Mass matrix for things that are not parameters
	float* parameterMatrix;							// Mass matrix for parameters
	float* parameterDerivitiveMatrix;				// Mass matrix for parameter derivitives
	float* parameterDerivitiveMatrixLocation;		// Location of the parameter derivitive matrix, for Runge Kutta 4th order
	
	std::vector<Layer*> layers;						// Layers of the neural network

	NeuralNetwork(uint32_t inputMatrixSize, float*& inputMatrix)
	{
		// user passes in the input size and a reference to the input matrix for use outside the class
		this->inputMatrixSize = inputMatrixSize;
		this->inputMatrix = new float[inputMatrixSize];
		inputMatrix = this->inputMatrix;
	}
	
	~NeuralNetwork()
	{
		// deletes the input, outputDerivative, the layers, and mass matrixes
		delete[] inputMatrix;
		delete[] outputDerivativeMatrix;
		for (Layer* layer : layers)
			delete layer;

		delete[] computationMatrix;
		delete[] parameterMatrix;
		delete[] parameterDerivitiveMatrix;
	}

	void AddLayer(Layer* layer)
	{
		// adds a layer to the neural network, user uses the new operator to create the layer and the class takes ownership
		layers.emplace_back(layer);
	}

	void Initialize(float*& outputMatrix, float*& outputDerivativeMatrix, float*& inputDerivativeMatrix)
	{
		// first creates outputDerivativeMatrix using the last layer and updates the user reference for use outside the class
		// next updates the input sizes of the layers, then loads the layer specs, then creates the mass matrixes
		// finally, updates the layer variables bases on the mass matrixes and updates outputMatrix and inputDerivativeMatrix for use outside the class
		
		std::vector<ComputationInfo> ComputationSpecs;
		std::vector<ParameterInfo> ParameterSpecs;

		this->outputDerivativeMatrix = new float[layers.back()->GetOutputMatrixSize()];
		outputDerivativeMatrix = this->outputDerivativeMatrix;

		layers[0]->AssignInputMatrixSize(inputMatrixSize);
		layers[0]->LoadLayerSpecs(ComputationSpecs, ParameterSpecs);
		for (uint32_t layer = 1; layer < layers.size(); layer++)
		{
			layers[layer]->AssignInputMatrixSize(layers[layer - 1]->GetOutputMatrixSize());
			layers[layer]->LoadLayerSpecs(ComputationSpecs, ParameterSpecs);
		}

		computationMatrixSize = 0;
		for (ComputationInfo& computationInfo : ComputationSpecs)
			computationMatrixSize += computationInfo.matrixSize;
		
		parameterMatrixSize = 0;
		for (ParameterInfo& parameterInfo : ParameterSpecs)
		{
			*parameterInfo.displacement = parameterMatrixSize;
			parameterMatrixSize += parameterInfo.matrixSize;
		}
		
		// 4 times the number of parameters for Runge Kutta 4th order
		// mass matrixes allow for (simple clearing of data by using memset), (simple mass matrix initialization using cpuGenerateUniform), and (simple adding using cpuSaxpy)
		parameterDerivitiveMatrixSize = parameterMatrixSize << 2;
		computationMatrix = new float[computationMatrixSize];
		parameterMatrix = new float[parameterMatrixSize];
		parameterDerivitiveMatrix = new float[parameterDerivitiveMatrixSize];
		parameterDerivitiveMatrixLocation = parameterDerivitiveMatrix;
		parameterDerivitiveMatrixDisplacement = 0;
		memset(parameterDerivitiveMatrix, 0, parameterDerivitiveMatrixSize * sizeof(float));
		cpuGenerateUniform(parameterMatrix, parameterMatrixSize, -1, 1);

		float* computationMatrixIndex = computationMatrix;
		for (ComputationInfo& computationInfo : ComputationSpecs)
		{
			*computationInfo.matrix = computationMatrixIndex;
			computationMatrixIndex += computationInfo.matrixSize;
		}

		float* dynamicMatrixLocation = parameterMatrix;
		for (ParameterInfo& parameterInfo : ParameterSpecs)
		{
			*parameterInfo.matrix = dynamicMatrixLocation;
			dynamicMatrixLocation += parameterInfo.matrixSize;
		}

		outputMatrix = layers.back()->GetOutputMatrix();
		inputDerivativeMatrix = layers[0]->GetInputDerivativeMatrix();

		layers.back()->AssignOutputDerivativeMatrix(outputDerivativeMatrix);
		layers.back()->parameterDerivitiveMatrixPointer = &parameterDerivitiveMatrixLocation;
		for (uint32_t i = layers.size() - 1; i--;)
		{
			layers[i + 1]->AssignInputMatrix(layers[i]->GetOutputMatrix());
			layers[i]->AssignOutputDerivativeMatrix(layers[i + 1]->GetInputDerivativeMatrix());
			layers[i]->parameterDerivitiveMatrixPointer = &parameterDerivitiveMatrixLocation;
		}
		layers[0]->AssignInputMatrix(inputMatrix);
	}

	void ForwardPropagate()
	{
		for (auto& layer : layers)
			layer->ForwardPropagate();
	}

	void BackPropagate(float scalar)
	{
		for (uint32_t i = layers.size(); i--;)
			layers[i]->BackPropagate();
		cpuSaxpy(parameterMatrixSize, &scalar, parameterDerivitiveMatrix, 1, parameterMatrix, 1);
		memset(parameterDerivitiveMatrix, 0, parameterDerivitiveMatrixSize * sizeof(float));
	}

	void Print()
	{
		for (auto& layer : layers)
			layer->Print();
		//printf("\n\n");
	}
};