// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include<stdlib.h>
// extern "C" c++中编译c格式的函数 ,如果用c语言编译就不需要(文件后缀名.c)
// __declspec(dllexport) 函数导出到库中
#include <stdio.h>

#ifdef __cplusplus //C++
#define XEXT extern "C"
#else
#define XEXT
#endif
// 判断是否是windows WIN32 _WIN32 



#ifdef _WIN32 // 包含win32和win64
#define XLIB XEXT __declspec(dllexport)
#else			// Mac Linux
#define XLIB XEXT
#endif

XLIB float* VST(int *arr, int width,int height)
{

	int count = width * height;
	static float *pA=NULL ;
	if (pA != NULL)
		free(pA);

	pA = (float*)malloc(count * sizeof(double));
	for (int i = 0; i < count; ++i) {
		pA[i]= arr[i];
		

	}
	return pA;
	
}