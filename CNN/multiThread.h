#pragma once
#ifndef _MULTITHREAD_H_
#define _MULTITHREAD_H_

#include "MyCNN.hpp"
#include "Win_Util.h"

///以下是对多线程执行的支持模块
typedef struct {
	MyCNN *cnn;
	int index;	//序号索引，可用作标记功能
} MyCNN_Index;

enum {
	Load_Test = 1,
	Mutate_Half = 2
};
extern volatile bool thread_state;//子线程状态

//等待子线程结束
void waitForFinish();

//MyCNN类的一个多线程辅助计算方法
RETURN_TYPE MyCNN_Thread(void* para);

#ifndef _MSC_VER
#include"multiThread.cpp"
#endif

#endif //_MYCNN_HPP_