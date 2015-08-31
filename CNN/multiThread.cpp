#include "multiThread.h"

volatile bool thread_state = 0;//子线程状态
//等待子线程结束
void waitForFinish()
{
	while (!thread_state)
		Sleep(10);
}

//MyCNN类的一个多线程辅助计算方法
RETURN_TYPE MyCNN_Thread(void* para)
{
	thread_state = 0;	//标记未完成
	MyCNN_Index cnn_index = *(MyCNN_Index*)para;
	if (Param::multiThread) {
		switch (cnn_index.index) {
		case Load_Test: //载入测试集
			cnn_index.cnn->loadTest(Param::testFile, Param::testSetNum, Param::testHaveTag);
			break;
		case Mutate_Half:
			int popSize = cnn_index.cnn->getPopSize(), i;
			for (i = popSize / 2; i < popSize; ++i)
				cnn_index.cnn->mutate(i);
			break;
		}
	}
	thread_state = 1;	//标记已完成
	return RETURN_VALUE;
}
