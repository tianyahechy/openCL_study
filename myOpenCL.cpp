#include "myOpenCL.h"
myOpenCL::myOpenCL(std::string strOpenCLFileName, std::string strOpenCLKernalEntry)
{
	_strOpenCLFileName = strOpenCLFileName;
	_strOpenCLKernalEntry = strOpenCLKernalEntry;
	_theContext = this->createContext();
	_commandQueue = this->createCommandQueue(_theContext, _device);
	_theProgram = this->createProgram(_strOpenCLFileName.c_str());
	//����opencl�ں�
	_theKernel = clCreateKernel(_theProgram, _strOpenCLKernalEntry.c_str(), NULL);
}

myOpenCL::~myOpenCL()
{
	if (_commandQueue != 0)
	{
		clReleaseCommandQueue(_commandQueue);
	}

	if (_theKernel != 0)
	{
		clReleaseKernel(_theKernel);
	}
	if (_theProgram != 0)
	{
		clReleaseProgram(_theProgram);
	}
	if (_theContext != 0)
	{
		clReleaseContext(_theContext);
	}
}

//Ϊcpuƽ̨����������
cl_context myOpenCL::createContext()
{
	cl_platform_id firstPlatformId = 0;
	cl_uint numPlatforms = 0;
	//����ѡ���һ��ƽ̨
	cl_int errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		std::cerr << "Failed to find any opencl platforms." << std::endl;
		return NULL;
	}
	else
	{
		std::cout << "��openclƽ̨" << std::endl;
	}

	//����ƽ̨��һ�������ģ�����ͼ����һ��gpu�ģ����û�еĻ����ʹ���cpu��
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};
	cl_context context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
		std::cout << "���ܴ���gpu������ ������CPU..." << std::endl;
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "���ܴ���opencl gpu����cpu������";
			return NULL;
		}
		else
		{
			std::cout << "�ܴ���cpu������" << std::endl;
		}
	}
	else
	{
		std::cout << "�ܴ���gpu������" << std::endl;
	}
	return context;
}
//ѡ���һ�������豸��������һ���������
cl_command_queue myOpenCL::createCommandQueue(cl_context context, cl_device_id & device)
{
	size_t deviceBufferSize = -1;
	cl_int errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "���ܻ�ȡ�豸�����С" << std::endl;
		return NULL;
	}
	else
	{
		std::cout << "�ɹ���ȡ�豸�����С" << std::endl;
	}

	//Ϊ�豸�������ռ�
	cl_device_id * devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "���ܵõ��豸id����" << std::endl;
		return NULL;
	}
	else
	{
		std::cout << "���Եõ��豸id����" << std::endl;
	}

	//����ֻѡ���һ�����õ��豸���ڸ��豸����һ���������.�������������ڽ�������Ҫִ�е��ں��Ŷӣ������ؽ��
	cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
	if (commandQueue == NULL)
	{
		std::cerr << "����Ϊ�豸0�����������" << std::endl;
		return NULL;
	}
	else
	{
		std::cout << "��Ϊ�豸0�����������" << std::endl;
	}
	device = devices[0];
	delete[] devices;
	return commandQueue;

}

//�Ӵ��̼����ں�Դ�ļ������͹���һ���������
cl_program myOpenCL::createProgram( const char* fileName)
{
	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "���ܴ��ļ�" << fileName << std::endl;
		return NULL;
	}
	else
	{
		std::cout << "�ɹ����ļ�" << fileName << std::endl;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();
	std::string srcStdStr = oss.str();
	const char * srcStr = srcStdStr.c_str();
	//����������� 
	cl_program program = clCreateProgramWithSource(_theContext, 1, (const char**)&srcStr, NULL, NULL);
	if (program == NULL)
	{
		std::cerr << "���ܴ�Դ�ļ�����opencl�������" << std::endl;
		return NULL;
	}
	else
	{
		std::cout << "�ܴ�Դ�ļ�����opencl�������" << std::endl;
	}

	//�����ں�Դ��
	cl_int errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		//�жϴ���ԭ��
		char buildLog[16384];
		clGetProgramBuildInfo(program, _device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
		std::cerr << "�ں˴���:" << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(program);
		return NULL;
	}
	else
	{
		std::cout << "�����ں�Դ��ɹ�" << std::endl;
	}
	return program;
}

//�����ڴ�������豸�ڴ��У��������ں˺���ֱ�ӷ���
bool myOpenCL::createMemObjects(cl_mem memObjects[3], float * a, float *b)
{
	memObjects[0] = clCreateBuffer(_theContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, a, NULL);
	memObjects[1] = clCreateBuffer(_theContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, b, NULL);
	memObjects[2] = clCreateBuffer(_theContext, CL_MEM_READ_WRITE, sizeof(float) * ARRAY_SIZE, NULL, NULL);
	if (memObjects[0] == NULL ||
		memObjects[1] == NULL ||
		memObjects[2] == NULL
		)
	{
		std::cerr << "���󴴽��ڴ����" << std::endl;
		return false;
	}
	return true;
}

//�����Դ
void myOpenCL::cleanUp( cl_mem memObjects[3])
{
	for (size_t i = 0; i < 3; i++)
	{
		if (memObjects[i] != 0)
		{
			clReleaseMemObject(memObjects[i]);
		}
	}

}

//�����豸������ 
cl_context myOpenCL::getContext()
{
	return _theContext;
}

//�����ں˲���
cl_int myOpenCL::setKernelParameter( int id, cl_mem theData)
{
	cl_int errNum = clSetKernelArg(_theKernel, id, sizeof(cl_mem), &theData);
	return errNum;
}

//ʹ���������ʹ�����豸��ִ�е��ں��Ŷ�
cl_int myOpenCL::setKernalQueue(size_t* globalWorkSize, size_t* localWorkSize)
{
	cl_int errNum = clEnqueueNDRangeKernel(_commandQueue, _theKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	return errNum;
}
//���ں˶��ؽ��
cl_int myOpenCL::readResult(cl_mem memObject, float * result)
{
	cl_int errNum = clEnqueueReadBuffer(_commandQueue, memObject, CL_TRUE, 0, ARRAY_SIZE * sizeof(float), result, 0, NULL, NULL);
	return errNum;
}