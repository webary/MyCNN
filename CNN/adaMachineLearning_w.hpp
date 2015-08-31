/** 智能自适应机器学习矩阵模版
	最后编辑时间: 2015.4.2  21:00
    文档说明: 对需要处理的数据集进行演化,以找到合适的路径实现机器学习
    变量含义: mlDimension——进入模板的数据维数;numPerCol——模板每列个数,
        根据Dimension的增大自动增长;chromLength——个体长度,即路径长度;
        popSize——同时进行演化的种群大小；MaxChromLength——最大个体长度;
        MaxGen——最大演化代数;Exp——误差计算指数;dataLen数据集个数(长度);
        bestChrom——最好个体;peout——个体在通过路径后的输出值;dataSet——数据集
    本次更新内容:
		2015:
		(8.24)将保存训练集和测试集的静态数组改为vector
		(7.26)将最大个体长度增至12,让矩阵输出4位0-1二进制数表示0-15
		(3.31)矩阵左边加入神经网络(一层神经元),模板内部数据范围由0~255变为0~1
		2014:
		(11.22)规范命名法则:
			popsize=>popSize,individual=>Individual,chromlength=>chromLength,
			fitnessvalue()=>fitnessValue(),findbest()=>findBest()
		(10.28)修改部分表达式，将一些固定参数改为可调（如OP_NUMS）
		(6.08)改为C++类的形式封装
		(5.14)更改需求,以适应不同情况下的机器学习
		(4.10)创建多线程,在出现询问对话框时后台仍然演化
		(3.24)解决演化速度有时过慢,无法由用户决定停止的问题
    @Author: 闻波(webary)
**/
#pragma once
#ifndef  _ADAMACHINELEARNING_HPP_
#define  _ADAMACHINELEARNING_HPP_

#define _USE_MATH_DEFINES	//使用math.h里的一些常数定义
#include <iostream>
#include <string>		//getline()函数
#include <fstream>		//getline()函数
#include <iomanip>		//setw(n)
#include <vector>
#include <cstdlib>		//rand()函数
#include <cmath>		//数学函数
#include <ctime>
#include <cstring>
#include <windows.h>
#include <process.h>	//_beginthread()函数
#include "math_util.hpp"
#include "Win_Util.h"
using namespace std;
typedef const char cchar;

#ifdef __AFXWIN_H__
extern int testOut[1000];
extern char ml_info[10000];
#endif
extern int flag_stop;

typedef unsigned uint;
typedef vector<float> vectorF;
class Ada_ML : virtual public Math_Util {
protected:
	int mlDimension, numPerCol, col, chromLength;
	const static int OP_NUMS = 8, OP_NUMSx3 = 3*OP_NUMS;	//模版操作符个数
	const static int MaxDimension = 300;//定义最大能接受的数据输入维数
	const static int MaxCol = 15;		//定义最大列数
	const static int popSize = 3;		//种群规模
	const static int MaxChromLength = (MaxDimension+OP_NUMSx3-1)/OP_NUMSx3*OP_NUMSx3*MaxCol+3*10;
	const static int MaxGen = 100000000;//演化学习的最大演化代数
	const static int Exp = 2;			//计算指数
	const static int Max_MatInput = (MaxDimension+OP_NUMSx3-1)/OP_NUMSx3 * OP_NUMS;//矩阵输入最大数
	int outPutModels;	//输出模块数
	typedef int ModelMatrixType;		//模板矩阵中值的类型
	ModelMatrixType PE[Max_MatInput][MaxCol];//模板矩阵的值
	int best,bestChrom[MaxChromLength];	//最好个体的索引,最好个体
	int dataLen, **PEout;		//数据集个数,总个数,模板输出数组

	typedef struct {
		vectorF data;			//输入数据部分
		int tag;				//输入标签(1 or 0 or ...)
	} S_Input;					//输入类型
	vector<S_Input> matTrain;	//训练集-包含若干训练数据和标签信息
	vector<S_Input> matTest;	//测试集-包含若干训练数据(也可包含标签信息)
	//typedef float DataType;	//模版能够接受的数据类型
	//DataType dataSet[5000][MaxDimension+1], testSet[5000][MaxDimension];//数据集，测试集
	float F,CR;	//变异系数,交叉因子
	float wait,permitError;//等待时间,允许误差
	typedef struct _Individual {
		unsigned char chrom[MaxChromLength];	//注意此处定义为char类型的必要性(内存占用更小)
		float ww[Max_MatInput][MaxDimension+1];	//当前已用部分为[0~mlDimension-1][0~mlDimension]
		bool  wwOut[Max_MatInput];
		double fitness;
	} Individual;
	Individual pop[popSize], cBest;
	int popRand[3];	//用于变异前产生几个随机个体
	//记录是否保存所有个体的误差，保存当前最好路径，保存最好路径，是否必须相等
	bool b_saveTmp, b_savePre, b_saveBest, b_mustEqual;
#ifdef __AFXWIN_H__
	CStatusBar *pS;		//MFC主窗口的状态栏指针
#endif

public:
	Ada_ML(int MLDimension,bool _mustEqual=1,int _col=6);
	virtual ~Ada_ML();
	//开始机器学习
	virtual void startLearn(int len=0,double permit_error=8,cchar *readFromFile=0,bool b_saveData=false);
#ifdef __AFXWIN_H__
	//在MFC中修改状态栏，显示实时学习进度
	virtual void startLearn(int len=0,CStatusBar *p,double permit_error=8,cchar* readFromFile=0,bool b_saveData=false);
#endif
	//从文件载入训练集
	int loadTrainSet(const string &fileName, int size = -1);
	//从文件载入测试集
	int loadTestSet(const string &fileName, int size=-1, bool haveTag=0);
	//得到测试结果，保存到int testOut[]数组中
	double getTestOut(const string &fileName, int len = -1,int print01Ratio = -1);
	//返回指定数据dataRow由特定个体PE模板核的输出
	int getPEOut(Individual& indiv, const vectorF &dataRow);
	//返回指定数据由最好个体PE模板核得到的输出
	int getBestPEOut(const vectorF &dataRow);
	//返回由个体indiv的神经网络输出进入矩阵后的输出
	int getMatOut(const Individual& indiv);
	//返回最好个体
	Individual& getBest() {
		return cBest;
	}
	//返回PE模板核列数
	const int& getCol()const {
		return col;
	}
	//返回个体长度
	const int& getChromlen()const {
		return chromLength;
	}
	static int getPopSize() {
		return popSize;
	}
	int getTestLen()const {
		return matTest.size();
	}
	//设置机器学习的参数
	void setML(int MLDimension,bool _mustEqual=1,int _col=8);
	//设置询问等待时间和模板核的列数
	void setWaitCol(double wt,int _col=0);
	//设置输出模块数
	void setOutputModels(int outModels);
	//设置部分数据是否保存
	void setSave(bool saveBest = 0,bool saveTmp = 0,bool savePre = 0);
	//保存某个个体
	void savePopMatrix(const char* file,int index,ios::openmode mode=ios::out);
protected:
	//从文件载入输入数据
	int loadInputData(const string &fileName, int size, vector<S_Input> &dataVec, bool haveTag);
	//从文件把数据保存到数组
	int  loadBest(cchar *filePath);
	//初始化机器学习模块
	void init(cchar *readFromFile = 0,bool b_saveData = false);
	//对X,Y进行模板核指定操作进行运算
	int  operationOnXY(int x,int y,int op);
	//计算第ii个个体的适应值
	void fitnessValue(Individual& indiv, int ii);
	//找到最好个体
	int  findBest();
	//高斯和多点方式变异个体的矩阵部分和神经网络部分
	void mutateMatrixNNByGauss(Individual& tmp);
	//在变异前产生几个不同于ii随机个体用于交叉步骤
	void generatePopRand(int ii);
	//差分方式变异个体的矩阵部分和神经网络部分
	void mutateMatrixNNByDE(Individual& tmp,int ii,int *_popRand=0);
	//个体变异并选择新一代个体,要求种群最少需要3个个体
	void mutate(int ii,int *_popRand=0);
	//变异后从ii号个体与indiv中选择优胜个体进入下一代
	void selectPop(Individual& indiv,int ii);
	//计算出神经网络第ii个个体对第row组数据的输出
	void getNNOut(Individual& indiv,const vectorF &dataRow);
	//将矩阵中个体某一维变为随机范围内的值
	int getChromRandValue(int i);
};

#ifdef __cplusplus
extern "C" {
//创建工作者线程进行后台计算和判断
RETURN_TYPE waiting(void* tt);
} 
#endif

#ifndef _MSC_VER
#include"adaMachineLearning_w.cpp"
#endif

#endif // _ADAMACHINELEARNING_HPP_
