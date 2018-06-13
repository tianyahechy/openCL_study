
#include "myOpenCL.h"

const int ARRAY_SIZE = 1000000;

int main(int argc, char ** argv)
{
	std::string strOpenCLFileName = "test.cl";
	std::string strOpenCLKernalEntry = "hello_kernel";		
	int sizeOfInputType = 2;
	int sizeOfInputObject = ARRAY_SIZE;
	int sizeOfEachInputUnit = sizeof(float);
	std::vector<std::vector<float>> inputVec2;	
	//设定各单元数值
	inputVec2.clear();
	inputVec2.resize(sizeOfInputType);
	for (size_t j = 0; j < sizeOfInputType; j++)
	{
		inputVec2[j].resize(sizeOfInputObject);
	}

	for (size_t i = 0; i < sizeOfInputObject; i++)
	{
		inputVec2[0][i] = (float)i;
		inputVec2[1][i] = (float)(i * 2);
	}

	int sizeOfOutputType = 1;
	int sizeOfOutputObject = ARRAY_SIZE;
	int sizeOfEachOutputUnit = sizeof(float);
	std::vector<std::vector<float>> outputVec2;
	outputVec2.clear();
	outputVec2.resize(sizeOfOutputType);
	for (size_t j = 0; j < sizeOfOutputType; j++)
	{
		outputVec2[j].resize(sizeOfOutputObject);
	}

	myOpenCL theOpenCL(strOpenCLFileName,
		strOpenCLKernalEntry,
		sizeOfInputType,
		sizeOfInputObject,
		sizeOfEachInputUnit,
		inputVec2,
		sizeOfOutputType,
		sizeOfOutputObject,
		sizeOfEachOutputUnit,
		outputVec2);

	theOpenCL.process();
	//输出结果
	std::vector<std::vector<float>>  resultVec = theOpenCL.getResult();
	int sizeOfResult = resultVec.size();
	for (size_t j = 0; j < sizeOfResult; j++)
	{
		for (size_t i = 0; i < resultVec[j].size(); i++)
		{
			if (i % 10 == 0)
			{
				std::cout << std::endl;
			}
			float theResult = resultVec[j][i];
			std::cout << theResult << ",";
		}
	
	}

	return 0;
}