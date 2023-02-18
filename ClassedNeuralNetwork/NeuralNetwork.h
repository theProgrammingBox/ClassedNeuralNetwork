#pragma once
#include "Layer.h"

class NeuralNetwork
{
public:
	std::vector<Layer*> layers;
	float* inputMatrix;
	float* outputDerivativeMatrix;

	struct DynamicLayerSpec
	{
		uint32_t size;
		float* matrix;
	};

	NeuralNetwork(uint32_t input, float* inputMatrix)
	{
		this->inputMatrix = new float[input];
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
			/*layers[i];
			layers[i + 1];*/
		}
	}

	void Print()
	{
		for (auto& layer : layers)
			layer->Print();
	}
};