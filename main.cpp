
#include "myOpenCL.h"
#include <vector>

int main(int argc, char ** argv)
{
	std::string strFileName = "test.cl";
	std::string strOpenCLKernalEntry = "hello_kernel";
	myOpenCL theOpenCL(strFileName, strOpenCLKernalEntry);

	int objectSize = 3;
	int numberOfEachObject = ARRAY_SIZE;
	std::vector<cl_mem> memObjectVector;
	memObjectVector.clear();
	memObjectVector.resize(objectSize);
	for (size_t i = 0; i < objectSize; i++)
	{
		memObjectVector[i] = 0;
	}
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
	theOpenCL.createMemObjects(&memObjectVector[0], &computeVector[0][0], &computeVector[1][0]);

	//�����ں˲���
	for (size_t i = 0; i < objectSize; i++)
	{
		theOpenCL.setKernelParameter(i, memObjectVector[i]);
	}
	//ʹ���������ʹ�����豸��ִ�е��ں��Ŷ�
	size_t globalWorkSize[1] = { ARRAY_SIZE };
	size_t localWorkSize[1] = { 1 };
	theOpenCL.setKernalQueue(globalWorkSize, localWorkSize);
	//���ں˶��ؽ��
	theOpenCL.readResult(memObjectVector[2], &computeVector[2][0]);
	for (size_t i = 0; i < ARRAY_SIZE; i++)
	{
		if ( i % 10 == 0 )
		{
			std::cout << std::endl;
		}
		std::cout << computeVector[2][i] << ",";
	}

	theOpenCL.cleanUp(&memObjectVector[0]);

	return 0;
}