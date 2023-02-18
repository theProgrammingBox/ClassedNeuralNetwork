#pragma once
#include "Layer.h"

class NeuralNetwork
{
public:
	float* inputMatrix;
	float* outputDerivativeMatrix;
	std::vector<Layer*> layers;

	NeuralNetwork(uint32_t inputMatrixSize, float* inputMatrix)
	{
		this->inputMatrix = new float[inputMatrixSize];
		inputMatrix = this->inputMatrix;
	}
	
	~NeuralNetwork()
	{
		delete[] inputMatrix;
		delete[] outputDerivativeMatrix;
		for (auto& layer : layers)
			delete layer;
	}

	void AddLayer(Layer* layer)
	{
		layers.emplace_back(layer);
	}

	void Initialize(float* outputMatrix, float* outputDerivativeMatrix, float* inputDerivativeMatrix)
	{
		std::vector<DynamicLayerSpec> dynamicLayerSpecs;

		for (uint32_t i = layers.size() - 1; i--;)
		{
			layers[i + 1]->AssignInputMatrix(layers[i]->GetOutputMatrix(), &dynamicLayerSpecs);
			/*layers[i];*/
		}
	}

	void Print()
	{
		for (auto& layer : layers)
			layer->Print();
	}
};