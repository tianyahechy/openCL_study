#include "myOpenCL.h"
myOpenCL::myOpenCL(std::string strOpenCLFileName,
	std::string strOpenCLKernalEntry,
	int objectSize,
	int numberOfEachObject,
	int sizeOfEachUnit,
	std::vector<std::vector<float>> inputVec2)
{
	_strOpenCLFileName = strOpenCLFileName;
	_strOpenCLKernalEntry = strOpenCLKernalEntry;
	_objectSize = objectSize;
	_numberOfEachObject = numberOfEachObject;
	_sizeOfEachUnit = sizeOfEachUnit;
	_inputVec2 = inputVec2;
	_theContext = NULL;
	_commandQueue = NULL;
	_theProgram = NULL;
	_theKernel = NULL;

}

myOpenCL::~myOpenCL()
{
	_inputVec2.clear();
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

//为cpu平台创建上下文
cl_context myOpenCL::createContext()
{
	cl_platform_id firstPlatformId = 0;
	cl_uint numPlatforms = 0;
	//这里选择第一个平台
	cl_int errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	//创建平台的一个上下文，先试图创建一个gpu的，如果没有的话，就创建cpu的
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};
	cl_context context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
	}
	return context;
}
//选择第一个可用设备，并创建一个命令队列
cl_command_queue myOpenCL::createCommandQueue(cl_context context, cl_device_id & device)
{
	size_t deviceBufferSize = -1;
	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	//为设备缓存分配空间
	cl_device_id * devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	//这里只选择第一个可用的设备，在该设备创建一个命令队列.这个命令队列用于将程序中要执行的内核排队，并读回结果
	cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
	
	device = devices[0];
	delete[] devices;
	return commandQueue;
}

//从磁盘加载内核源文件创建和构建一个程序对象
cl_program myOpenCL::createProgram( const char* fileName)
{
	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "不能打开文件" << fileName << std::endl;
		return NULL;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();
	std::string srcStdStr = oss.str();
	const char * srcStr = srcStdStr.c_str();
	//创建程序对象 
	cl_program program = clCreateProgramWithSource(_theContext, 1, (const char**)&srcStr, NULL, NULL);
	//编译内核源码
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	return program;
}

//返回设备上下文 
cl_context myOpenCL::getContext()
{
	return _theContext;
}

//建立内核参数
cl_int myOpenCL::setKernelParameter( int id, cl_mem theData)
{
	cl_int errNum = clSetKernelArg(_theKernel, id, sizeof(cl_mem), &theData);
	return errNum;
}

//使用命令队列使将在设备上执行的内核排队
cl_int myOpenCL::setKernalQueue(size_t* globalWorkSize, size_t* localWorkSize)
{
	cl_int errNum = clEnqueueNDRangeKernel(_commandQueue, _theKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	return errNum;
}
//从内核读回结果
cl_int myOpenCL::readResult(cl_mem memObject, float * result)
{
	cl_int errNum = clEnqueueReadBuffer(_commandQueue, memObject, CL_TRUE, 0, _numberOfEachObject * _sizeOfEachUnit, result, 0, NULL, NULL);
	return errNum;
}

//处理全过程
void myOpenCL::process()
{
	_theContext = this->createContext();
	_commandQueue = this->createCommandQueue(_theContext, _device);
	_theProgram = this->createProgram(_strOpenCLFileName.c_str());
	//创建opencl内核
	_theKernel = clCreateKernel(_theProgram, _strOpenCLKernalEntry.c_str(), NULL);

	//读写vector,用以在分配内存时判断只读还是读写
	//设定前几个只读，最后一个读写
	std::vector<int> readWriteVector;
	readWriteVector.clear();
	readWriteVector.resize(_objectSize);
	for (size_t i = 0; i < _objectSize - 1; i++)
	{
		readWriteVector[i] = readWrite::readOnly;
	}
	readWriteVector[_objectSize - 1] = readWrite::rw;

	std::vector<cl_mem> memObjectVector;
	memObjectVector.clear();
	memObjectVector.resize(_objectSize);
	for (size_t i = 0; i < _objectSize; i++)
	{
		memObjectVector[i] = 0;
	}

	cl_context theContext = this->getContext();
	//先读后写分配内存
	for (size_t i = 0; i < _objectSize; i++)
	{
		int readW = readWriteVector[i];
		if (readW == readWrite::readOnly)
		{
			memObjectVector[i] = clCreateBuffer(theContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
				sizeof(float) * _numberOfEachObject, &_inputVec2[i][0], NULL);
		}
		else if (readW == readWrite::rw)
		{
			memObjectVector[i] = clCreateBuffer(theContext, CL_MEM_READ_WRITE, _sizeOfEachUnit * _numberOfEachObject, NULL, NULL);
		}
	}

	//建立内核参数
	for (size_t i = 0; i < _objectSize; i++)
	{
		this->setKernelParameter(i, memObjectVector[i]);
	}
	//使用命令队列使将在设备上执行的内核排队
	size_t globalWorkSize[1] = { _numberOfEachObject };
	size_t localWorkSize[1] = { 1 };
	this->setKernalQueue(globalWorkSize, localWorkSize);
	//从内核读回结果
	this->readResult(memObjectVector[_objectSize - 1], &_inputVec2[_objectSize - 1][0]);
	memObjectVector.clear();
}

//返回结果
std::vector<float> myOpenCL::getResult()
{
	return _inputVec2[_objectSize - 1];
}