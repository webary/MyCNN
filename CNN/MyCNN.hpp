/*
 * Copyright (c) 2015
 * 闻波, webary, HBUT 
 * reference ：https://github.com/webary/MyCNN/
 * 	           http://www.cnblogs.com/webary/
 */
 
#pragma once
#ifndef _MYCNN_HPP_
#define _MYCNN_HPP_

#include "CNN_Base.hpp"
#include "adaMachineLearning_w.hpp"
#include "Param.hpp"

/**
@usage:将卷积网络基类CNN_Base与矩阵网络类Ada_ML结合起来构造完整的卷积神经网络模型
**/
class MyCNN : public Ada_ML, public CNN_Base {
	static const unsigned CNNPopSize = popSize;//卷积网络个体数
	static double F_CNN, F_Matrix;				//卷积网络的变异因子,矩阵网络的变异因子
	uint bestPopIndex;							//最好个体的索引
public:
	//右边网络的输入由CNN的输出决定
	MyCNN(uint modelNums = 1, uint _inputSize = 32, bool _mustEqual = 1);
	virtual ~MyCNN();
	//开始卷积网络学习，把普通神经网络的学习作为其中一部分
	void startCNNLearn(double permit_error = 8, cchar *readFromFile = 0);
	//得到测试集的输出
	void getTestOut();
	//对比测试集的输出正确率---前提是已知每组测试数据的输出
	float compareTestOut();
	//更新层---仅仅用来测试运行速度
	void updateLayer()
	{
		for (uint j = 0; j < popSize; ++j) {
			mutate(j);
			updatePerLayer(cnnTrain[0].data, cnnPop[j]);	//更新卷积网络
		}
	}
	//保存最好个体的卷积网络可训练参数
	void saveBestCnnPop(const char* file);
	//保存最好个体的矩阵网络参数
	void saveBestMatPop(const char* file);
	//保存当前种群最好个体的可演化参数到文件中
	void saveBestToFile(const char* file);
	//从文件把载入备份好的数据
	void loadBestPop(cchar *filePath);
	//对整个个体执行变异,包括卷积网络部分和矩阵网络部分;设置为公有让子线程可访问
	void mutate(uint index);
protected:
	//使用高斯变异和多点变异构造新新个体---mutate的一种实现方式
	void mutateByGauss_MultiPoint(CNNIndividual& _cnnPop, Individual& _pop);
	//从卷积网络得到输出作为矩阵网络的输入,计算个体_cnnPop与_pop结合后的适应值
	float getFitValue(CNNIndividual& _cnnPop, Individual& _pop);
	//找到当前最好个体
	int findBest()
	{
		bestPopIndex = 0;
		for (uint i = 1; i < popSize; ++i)
			if (cnnPop[i].fitValue < cnnPop[bestPopIndex].fitValue)
				bestPopIndex = i;
		return bestPopIndex;
	}
};

#ifndef _MSC_VER
#include"MyCNN.cpp"
#endif

#endif //_MYCNN_HPP_
