#include "MyCNN.hpp"  //卷积网络主类
#include "multiThread.h" //多线程支持的相关函数声明

int main()
{
	cout << "start at " << Math_Util::getDateTime() << endl;
	SetText(FG_HL | FG_G | FG_B);
	try {
		//载入配置文件中的参数设置
		Param::loadINI("set.ini", "MyCNN");	//读取配置文件"set.ini"中的“MyCNN”节点
		MyCNN cnn(Param::outputNums, 28);	//输出模块数,输入图大小
		cout << "正在载入训练数据和测试数据...";
		time_t t_start = clock();
		if (Param::multiThread) { //如果开启了多线程,就让子线程读取测试集
			MyCNN_Index cnn_index = { &cnn,Load_Test };
			_beginthread(MyCNN_Thread, 0, &cnn_index);//启动子线程
			cnn.loadTrain(Param::trainFile, Param::trainSetNum);//载入训练集
			waitForFinish();
		}
		else { //如果没有开启多线程,就依次载入训练集和测试集
			cnn.loadTrain(Param::trainFile, Param::trainSetNum);//载入训练试集
			cnn.loadTest(Param::testFile, Param::testSetNum, Param::testHaveTag);//载入测试集
		}
		cout << "\r载入训练数据和测试数据耗时: " << (clock() - t_start) << "ms" << endl;
		t_start = clock();
		cnn.startCNNLearn(1, "BestPop.bak");  //让卷积网络开始学习
		cout << "\n演化共耗时: " << (clock() - t_start) / 1000.0 << " s" << endl;
		t_start = clock();
		cout << "对测试集的识别正确率为: " << cnn.compareTestOut() << endl; //输出用学习得到的规则对测试集的输出比对结果
		t_start = clock();
		cnn.printCSToFile("CS.txt"); //将训练的最好结果的各层参数值以及中间神经元的值输出到文件
	}
	catch (const char* str) { //捕获到已知异常
		cout << "\n--error:" << str << endl;
	}
	catch (...) { //捕获到未知异常
		cout << "\nOops, there are some jokes in the runtime \\(╯-╰)/" << endl;
	}
	return 0;
}
