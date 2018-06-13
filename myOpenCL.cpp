#include "myOpenCL.h"
myOpenCL::myOpenCL()
{
	_theContext = this->createContext();
}

myOpenCL::~myOpenCL()
{
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
cl_program myOpenCL::createProgram(cl_context context, cl_device_id device, const char* fileName)
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
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, NULL, NULL);
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
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
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
bool myOpenCL::createMemObjects(cl_context context, cl_mem memObjects[3], float * a, float *b)
{
	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, a, NULL);
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, b, NULL);
	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * ARRAY_SIZE, NULL, NULL);
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
void myOpenCL::cleanUp(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel theKernel, cl_mem memObjects[3])
{
	for (size_t i = 0; i < 3; i++)
	{
		if (memObjects[i] != 0)
		{
			clReleaseMemObject(memObjects[i]);
		}
	}
	if (commandQueue != 0)
	{
		clReleaseCommandQueue(commandQueue);
	}

	if (theKernel != 0)
	{
		clReleaseKernel(theKernel);
	}
	if (program != 0)
	{
		clReleaseProgram(program);
	}
	if (context != 0)
	{
		clReleaseContext(context);
	}
}

//�����豸������ 
cl_context myOpenCL::getContext()
{
	return _theContext;
}