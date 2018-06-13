
#include "myOpenCL.h"

int main(int argc, char ** argv)
{
	std::string fileName = "test.cl";
	cl_mem memObjects[3] = { 0, 0, 0 };
	myOpenCL theOpenCL;
	cl_context theContext = theOpenCL.createContext();
	cl_device_id  device = NULL;
	cl_command_queue theQueue = theOpenCL.createCommandQueue(theContext, device);
	cl_program theProgram = theOpenCL.createProgram(theContext, device, fileName.c_str());
	//����opencl�ں�
	cl_kernel theKernel = clCreateKernel(theProgram, "hello_kernel", NULL);
	if (theKernel == NULL )
	{
		std::cerr << "���ܴ����ں�" << std::endl;
		theOpenCL.cleanUp(theContext, theQueue, theProgram, theKernel, memObjects);
		return 1;
	}
	else
	{
		std::cout << "�ɹ������ں�" << std::endl;
	}

	float result[ARRAY_SIZE];
	float a[ARRAY_SIZE];
	float b[ARRAY_SIZE];
	for (size_t i = 0; i < ARRAY_SIZE; i++)
	{
		a[i] = (float)i;
		b[i] = (float)(i * 2);
	}
	if (!theOpenCL.createMemObjects(theContext, memObjects, a, b))
	{
		std::cout << "�����ڴ����ʧ��" << std::endl;
		theOpenCL.cleanUp(theContext, theQueue, theProgram, theKernel, memObjects);
		return 1;
	}
	else
	{
		std::cout << "�����ڴ����ɹ�" << std::endl;
	}

	//�����ں˲���
	cl_int errNum = clSetKernelArg(theKernel, 0, sizeof(cl_mem), &memObjects[0]);
	errNum |= clSetKernelArg(theKernel, 1, sizeof(cl_mem), &memObjects[1]);
	errNum |= clSetKernelArg(theKernel, 2, sizeof(cl_mem), &memObjects[2]);
	if ( errNum != CL_SUCCESS )
	{
		std::cerr << "���������ں˲���" << std::endl;
		theOpenCL.cleanUp(theContext, theQueue, theProgram, theKernel, memObjects);
		return 1;
	}
	else
	{
		std::cout << "��ȷ�����ں˲���" << std::endl;
	}

	//ʹ���������ʹ�����豸��ִ�е��ں��Ŷ�
	size_t globalWorkSize[1] = { ARRAY_SIZE };
	size_t localWorkSize[1] = { 1 };
	errNum = clEnqueueNDRangeKernel(theQueue, theKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (errNum != CL_SUCCESS )
	{
		std::cerr << "�����ִ�ж����ں�" << std::endl;
		theOpenCL.cleanUp(theContext, theQueue, theProgram, theKernel, memObjects);
		return 1;
	}
	else
	{
		std::cout << "������ȷ��ִ�ж����ں�" << std::endl;
	}

	//���ں˶��ؽ��
	errNum = clEnqueueReadBuffer(theQueue, memObjects[2], CL_TRUE, 0, ARRAY_SIZE * sizeof(float), result, 0, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "�����ȡ�������" << std::endl;
		theOpenCL.cleanUp(theContext, theQueue, theProgram, theKernel, memObjects);
		return 1;
	}
	for (size_t i = 0; i < ARRAY_SIZE; i++)
	{
		if ( i % 10 == 0 )
		{
			std::cout << std::endl;
		}
		std::cout << result[i] << ",";
	}

	theOpenCL.cleanUp(theContext, theQueue, theProgram, theKernel, memObjects);

	return 0;
}