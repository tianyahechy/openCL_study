
#include "myOpenCL.h"

const int ARRAY_SIZE = 1000000;

int main(int argc, char ** argv)
{
	std::string strFileName = "test.cl";
	std::string strOpenCLKernalEntry = "hello_kernel";
	int objectSize = 3;
	int numberOfEachObject = ARRAY_SIZE;

	//设定各单元数值
	std::vector<std::vector<float>> computeVector;
	computeVector.clear();
	computeVector.resize(objectSize);
	for (size_t j = 0; j < objectSize; j++)
	{
		computeVector[j].resize(numberOfEachObject);
	}

	for (size_t i = 0; i < numberOfEachObject; i++)
	{
		computeVector[0][i] = (float)i;
		computeVector[1][i] = (float)(i * 2);
	}

	myOpenCL theOpenCL(strFileName, strOpenCLKernalEntry,objectSize,numberOfEachObject,sizeof(float),computeVector);
	theOpenCL.process();
	//输出结果
	std::vector<float> resultVec = theOpenCL.getResult();
	int sizeOfResult = resultVec.size();
	for (size_t i = 0; i < sizeOfResult; i++)
	{
		if (i % 10 == 0)
		{
			std::cout << std::endl;
		}
		float theResult = resultVec[i];
		std::cout << theResult << ",";
	}

	return 0;
}