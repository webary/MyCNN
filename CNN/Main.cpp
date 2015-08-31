#include "MyCNN.hpp"
#include "multiThread.h"

int main()
{
	cout << "start at " << Math_Util::getDateTime() << endl;
	SetText(FG_HL | FG_G | FG_B);
	try {
		//载入配置文件中的参数设置
		Param::loadINI("set.ini", "MyCNN");
		MyCNN cnn(Param::outputNums, 28);	//输出模块数,输入图大小
		cout << "正在载入训练数据和测试数据...";
		time_t t_start = clock();
		if (Param::multiThread) { //如果开启了多线程,就让子线程读取测试集
			MyCNN_Index cnn_index = { &cnn,Load_Test };
			_beginthread(MyCNN_Thread, 0, &cnn_index);
			cnn.loadTrain(Param::trainFile, Param::trainSetNum);
			waitForFinish();
		}
		else {
			cnn.loadTrain(Param::trainFile, Param::trainSetNum);
			cnn.loadTest(Param::testFile, Param::testSetNum, Param::testHaveTag);
		}
		cout << "\r载入训练数据和测试数据耗时: " << (clock() - t_start) << "ms" << endl;
		t_start = clock();
		cnn.startCNNLearn(1, "BestPop.bak");
		cout << "\n演化共耗时: " << (clock() - t_start) / 1000.0 << " s" << endl;
		t_start = clock();
		cout << "对测试集的识别正确率为: " << cnn.compareTestOut() << endl;
		t_start = clock();
		cnn.printCSToFile("CS.txt");
	}
	catch (const char* str) {
		cout << "\n--error:" << str << endl;
	}
	catch (...) {
		cout << "\nOops, there are some jokes in the runtime \\(╯-╰)/" << endl;
	}
	return 0;
}
