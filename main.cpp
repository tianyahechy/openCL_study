
#include "myOpenCL.h"
#include <vector>

int main(int argc, char ** argv)
{
	std::string strFileName = "test.cl";
	std::string strOpenCLKernalEntry = "hello_kernel";
	int objectSize = 3;
	int numberOfEachObject = ARRAY_SIZE;
	std::vector<cl_mem> memObjectVector;
	memObjectVector.clear();
	memObjectVector.resize(objectSize);
	for (size_t i = 0; i < objectSize; i++)
	{
		memObjectVector[i] = 0;
	}
	myOpenCL theOpenCL(strFileName, strOpenCLKernalEntry);
	float result[ARRAY_SIZE];
	float a[ARRAY_SIZE];
	float b[ARRAY_SIZE];
	for (size_t i = 0; i < ARRAY_SIZE; i++)
	{
		a[i] = (float)i;
		b[i] = (float)(i * 2);
	}
	theOpenCL.createMemObjects(&memObjectVector[0], a, b);

	//�����ں˲���
	for (size_t i = 0; i < 3; i++)
	{
		theOpenCL.setKernelParameter(i, memObjectVector[i]);
	}
	//ʹ���������ʹ�����豸��ִ�е��ں��Ŷ�
	size_t globalWorkSize[1] = { ARRAY_SIZE };
	size_t localWorkSize[1] = { 1 };
	theOpenCL.setKernalQueue(globalWorkSize, localWorkSize);
	//���ں˶��ؽ��
	theOpenCL.readResult(memObjectVector[2], result);
	for (size_t i = 0; i < ARRAY_SIZE; i++)
	{
		if ( i % 10 == 0 )
		{
			std::cout << std::endl;
		}
		std::cout << result[i] << ",";
	}

	theOpenCL.cleanUp(&memObjectVector[0]);

	return 0;
}