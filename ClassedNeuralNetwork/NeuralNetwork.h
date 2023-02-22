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
	uint32_t rungeKuttaStep;
	float* computationMatrix;						// Mass matrix for things that are not parameters
	float* parameterMatrix;							// Mass matrix for parameters
	float* tempParameterMatrix;						// Temporary mass matrix for parameters, for Runge Kutta 4th order
	float* savedParameterMatrix;					// Saved mass matrix for parameters, for Runge Kutta 4th order
	float* parameterDerivitiveMatrix;				// Mass matrix for parameter derivitives
	
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
		delete[] tempParameterMatrix;
		delete[] savedParameterMatrix;
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

		std::vector<PartitionData> computationPartitionData;
		std::vector<PartitionData> parameterPartitionData;
		std::vector<PartitionData> parameterDerivitivePartitionData;

		this->outputDerivativeMatrix = new float[layers.back()->GetOutputMatrixSize()];
		outputDerivativeMatrix = this->outputDerivativeMatrix;

		layers[0]->AssignInputMatrixSize(inputMatrixSize);
		layers[0]->LoadLayerSpecs(computationPartitionData, parameterPartitionData, parameterDerivitivePartitionData);
		for (uint32_t layer = 1; layer < layers.size(); layer++)
		{
			layers[layer]->AssignInputMatrixSize(layers[layer - 1]->GetOutputMatrixSize());
			layers[layer]->LoadLayerSpecs(computationPartitionData, parameterPartitionData, parameterDerivitivePartitionData);
		}

		computationMatrixSize = 0;
		for (PartitionData& computationPartition : computationPartitionData)
			computationMatrixSize += computationPartition.matrixSize;
		
		parameterMatrixSize = 0;
		for (PartitionData& parameterPartition : parameterPartitionData)
			parameterMatrixSize += parameterPartition.matrixSize;
		
		rungeKuttaStep = 0;
		computationMatrix = new float[computationMatrixSize];
		parameterMatrix = new float[parameterMatrixSize];
		tempParameterMatrix = new float[parameterMatrixSize];
		savedParameterMatrix = new float[parameterMatrixSize];
		parameterDerivitiveMatrix = new float[parameterMatrixSize];
		cpuGenerateUniform(parameterMatrix, parameterMatrixSize, -1, 1);
		//memcpy(tempParameterMatrix, parameterMatrix, parameterMatrixSize * sizeof(float));

		float* computationMatrixIndex = computationMatrix;
		for (PartitionData& computationPartition : computationPartitionData)
		{
			*computationPartition.matrix = computationMatrixIndex;
			computationMatrixIndex += computationPartition.matrixSize;
		}

		float* parameterMatrixLocation = tempParameterMatrix;
		for (PartitionData& parameterPartition : parameterPartitionData)
		{
			*parameterPartition.matrix = parameterMatrixLocation;
			parameterMatrixLocation += parameterPartition.matrixSize;
		}

		float* parameterDerivitiveMatrixLocation = parameterDerivitiveMatrix;
		for (PartitionData& parameterDerivitivePartition : parameterDerivitivePartitionData)
		{
			*parameterDerivitivePartition.matrix = parameterDerivitiveMatrixLocation;
			parameterDerivitiveMatrixLocation += parameterDerivitivePartition.matrixSize;
		}

		outputMatrix = layers.back()->GetOutputMatrix();
		inputDerivativeMatrix = layers[0]->GetInputDerivativeMatrix();

		layers.back()->AssignOutputDerivativeMatrix(outputDerivativeMatrix);
		for (uint32_t i = layers.size() - 1; i--;)
		{
			layers[i + 1]->AssignInputMatrix(layers[i]->GetOutputMatrix());
			layers[i]->AssignOutputDerivativeMatrix(layers[i + 1]->GetInputDerivativeMatrix());
		}
		layers[0]->AssignInputMatrix(inputMatrix);
	}

	void ForwardPropagate()
	{
		memset(computationMatrix, 0, computationMatrixSize * sizeof(float));
		for (auto& layer : layers)
			layer->ForwardPropagate();
	}

	void BackPropagate()
	{
		if (rungeKuttaStep == 0)
			memcpy(savedParameterMatrix, parameterMatrix, parameterMatrixSize * sizeof(float));
		memcpy(tempParameterMatrix, savedParameterMatrix, parameterMatrixSize * sizeof(float));
		if (rungeKuttaStep != 0)
			cpuSaxpy(parameterMatrixSize, &GLOBAL::applied[rungeKuttaStep], parameterDerivitiveMatrix, 1, tempParameterMatrix, 1);
		memset(parameterDerivitiveMatrix, 0, parameterMatrixSize * sizeof(float));
		//memset(parameterDerivitiveMatrix, 0, parameterMatrixSize * sizeof(float));

		for (uint32_t i = layers.size(); i--;)
			layers[i]->BackPropagate();
		
		cpuSaxpy(parameterMatrixSize, &GLOBAL::summed[rungeKuttaStep], parameterDerivitiveMatrix, 1, parameterMatrix, 1);
		rungeKuttaStep *= ++rungeKuttaStep == 4;
		//cpuSaxpy(parameterMatrixSize, &GLOBAL::LEARNING_RATE, parameterDerivitiveMatrix, 1, tempParameterMatrix, 1);
	}

	void Print()
	{
		for (auto& layer : layers)
			layer->Print();
		printf("\n\n");
	}

	/*void Export(const char* fileName)
	{
		// first save the number of layers, then save their type/details, finally save parameterMatrix mass matrix
		std::ofstream file(fileName, std::ios::out | std::ios::binary);
		uint32_t numberOfLayers = layers.size();
		file.write((char*)&numberOfLayers, sizeof(uint32_t));
		for (auto& layer : layers)
			layer->Export(file);
		//
		file.write((char*)parameterMatrix, parameterMatrixSize * sizeof(float));
		file.close();
	}*/
};