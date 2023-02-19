#pragma once
#include "Layer.h"

class NeuralNetwork
{
public:
	uint32_t inputMatrixSize;
	float* inputMatrix;
	float* outputMatrix;
	float* outputDerivativeMatrix;
	float* inputDerivativeMatrix;
	
	uint32_t staticMatrixSize;
	uint32_t dynamicMatrixSize;
	uint32_t dynamicMatrixDerivitiveSize;
	uint32_t dynamicParamDerivitiveDisplacement;
	float* staticMatrix;
	float* dynamicParamMatrix;
	float* dynamicParamDerivitiveMatrix;
	float* currentDynamicParamDerivitiveLocation;
	
	std::vector<Layer*> layers;

	NeuralNetwork(uint32_t inputMatrixSize, float*& inputMatrix)
	{
		this->inputMatrixSize = inputMatrixSize;
		this->inputMatrix = new float[inputMatrixSize];
		inputMatrix = this->inputMatrix;
	}
	
	~NeuralNetwork()
	{
		delete[] inputMatrix;
		delete[] outputDerivativeMatrix;
		for (Layer* layer : layers)
			delete layer;

		delete[] staticMatrix;
		delete[] dynamicParamMatrix;
		delete[] dynamicParamDerivitiveMatrix;
	}

	void AddLayer(Layer* layer)
	{
		layers.emplace_back(layer);
	}

	void Initialize(float*& outputMatrix, float*& outputDerivativeMatrix, float*& inputDerivativeMatrix)
	{
		std::vector<StaticMatrixInfo> staticParams;
		std::vector<DynamicMatrixInfo> dynamicParams;

		outputDerivativeMatrix = new float[layers.back()->GetOutputMatrixSize()];

		for (uint32_t i = layers.size() - 1; i--;)
		{
			layers[i + 1]->AssignInputMatrixSize(layers[i]->GetOutputMatrixSize());
			layers[i + 1]->LoadLayerSpecs(
				staticParams,
				dynamicParams
			);
		}
		layers[0]->AssignInputMatrixSize(inputMatrixSize);
		layers[0]->LoadLayerSpecs(
			staticParams,
			dynamicParams
		);

		staticMatrixSize = 0;
		for (StaticMatrixInfo& matrixInfo : staticParams)
			staticMatrixSize += matrixInfo.matrixSize;
		
		dynamicMatrixSize = 0;
		for (DynamicMatrixInfo& matrixInfo : dynamicParams)
		{
			*matrixInfo.displacement = dynamicMatrixSize;
			dynamicMatrixSize += matrixInfo.matrixSize;
		}
		
		dynamicMatrixDerivitiveSize = dynamicMatrixSize << 2;
		dynamicParamDerivitiveDisplacement = 0;
		staticMatrix = new float[staticMatrixSize];
		dynamicParamMatrix = new float[dynamicMatrixSize];
		dynamicParamDerivitiveMatrix = new float[dynamicMatrixDerivitiveSize];
		currentDynamicParamDerivitiveLocation = dynamicParamDerivitiveMatrix;
		ResetDynamicParamDerivitiveMatrix();

		float* staticMatrixIndex = staticMatrix;
		for (StaticMatrixInfo& matrixInfo : staticParams)
		{
			*matrixInfo.matrix = staticMatrixIndex;
			staticMatrixIndex += matrixInfo.matrixSize;
		}

		float* dynamicMatrixLocation = dynamicParamMatrix;
		for (DynamicMatrixInfo& matrixInfo : dynamicParams)
		{
			*matrixInfo.matrix = dynamicMatrixLocation;
			dynamicMatrixLocation += matrixInfo.matrixSize;
		}

		outputMatrix = layers.back()->GetOutputMatrix();
		inputDerivativeMatrix = layers[0]->GetInputDerivativeMatrix();
		
		cpuGenerateUniform(dynamicParamMatrix, dynamicMatrixSize, -1, 1);

		layers.back()->AssignOutputDerivativeMatrix(outputDerivativeMatrix);
		for (uint32_t i = layers.size() - 1; i--;)
		{
			layers[i + 1]->AssignInputMatrix(layers[i]->GetOutputMatrix());
			layers[i + 1]->AssignOutputDerivativeMatrix(layers[i]->GetInputDerivativeMatrix());
			layers[i + 1]->dynamicDerivativeMatrixPointer = &currentDynamicParamDerivitiveLocation;
		}
		layers[0]->AssignInputMatrix(inputMatrix);
	}

	void ForwardPropagate()
	{
		for (auto& layer : layers)
			layer->ForwardPropagate();
	}

	void BackPropagate(float dt)
	{
		for (auto& layer : layers)
			layer->BackPropagate(dt);
	}

	void Print()
	{
		PrintMatrix(inputMatrix, 1, inputMatrixSize, "Input Matrix");
		for (auto& layer : layers)
			layer->Print();
	}

private:
	void ResetDynamicParamDerivitiveMatrix()
	{
		memset(dynamicParamDerivitiveMatrix, 0, dynamicMatrixDerivitiveSize);
	}
};