/** 卷积网络部分开始-->
	7.16: 加入多线程加速运算;加入从配置文件载入参数
          将卷积核共享去掉;将CNN的S5作为右边网络的输入
	7.15: 高斯变异中加入步长;加入自动备份和载入备份的功能
**/

#pragma once
#ifndef _CNN_BASE_HPP_
#define _CNN_BASE_HPP_

#include"math_util.hpp"
#include<iostream>
#include<fstream>
#include<iomanip>
#include<vector>
#include<cfloat>
using namespace std;
typedef unsigned uint;
typedef vector<float> vectorF,biasType;
typedef vector<vectorF> vectorF2D,kernelType;
typedef vector<vector<unsigned> >  vectorU2D;
typedef vector<vector<bool> >  vectorB2D;
/**
	卷积网络基类，只包含卷积网络部分的属性和操作
**/
class CNN_Base : public Math_Util{
protected:
    enum SS_Mode {SSM_Mean, SSM_Max, SSM_Min};	//亚采样模式.subSample Mode

    static const uint kernelNum 	= 75;//卷积核个数
    static const uint kernelSize 	= 5;//卷积核大小
    static const uint subSampleSize = 2;//亚采样核大小
    static const float Max_kernel,Min_kernel, Max_bias,Min_bias;	//卷积核和偏置的范围
    //输入图像允许的大小由卷积核大小和亚采样大小决定
    //static const uint inputSize = kernelSize*(subSampleSize*(subSampleSize+1)+1)-subSampleSize-1;
    uint inputSize;
    static const uint CS_NUM[6];		//每层feature Map数
    vector<vectorF2D> C1,S2,C3,S4,C5,S6;//C1~S6层
    typedef struct {
        vectorF2D data;					//卷积网络的输入图数据部分
        int tag;						//卷积网络的输入图标签(真or假 or ...)
    } S_CNNInput;						//卷积网络输入类型
    vector<S_CNNInput> cnnTrain;		//卷积网络训练集-包含若干图像的灰度数据和标签信息
    vector<S_CNNInput> cnnTest;			//卷积网络测试集-包含若干图像的灰度数据(也可包含标签信息)
    typedef struct {
        kernelType c_kernel[kernelNum];	//卷积核
        biasType bias;					//偏置参数
        float stepInGuass[2];			//高斯变异中的步长(分别控制卷积核和偏置)
        float fitValue;					//适应值
    } CNNIndividual;					//定义卷积网络中的个体类型,包含所有可训练参数的集合即为一个个体
    vector<CNNIndividual> cnnPop;		//卷积网络个体集合
public:
    CNN_Base(uint _inputSize=32,uint _popSize=4);
    virtual ~CNN_Base();
    //增加新的输入图
    void addTrain(const vectorF2D& grayData,int tag);
    //从文件载入训练数据
    void loadTrain(const char* file,int size=0);
    //从文件载入测试数据
    void loadTest(const char* file,int size=0,bool haveTag=0);
    //用inputData作为输入更新C1~S6每层feature Map的值
    void updatePerLayer(const vectorF2D& inputData, const CNNIndividual& indiv);
    //得到CNN的最后一层的输出
    template<typename T>
    void getCNNOut(T& output)
    {
        for(uint i=0; i<CS_NUM[4]; ++i)
            output[i] = C5[i][0][0];
    }
    //对可训练参数(卷积核和偏置)执行变异
    virtual void kernelBiasMutate();
    //将各层的特征图打印到文件中
    void printCSToFile(const char* file);
    //将当前最好个体的可演化参数保存到文件中
    virtual void saveBestToFile(const char* file)=0;
protected:
	//载入输入数据，被loadTrain和loadTest调用
	int loadInput(const char* file,int size,vector<S_CNNInput> &dataVec,bool haveTag);
    //构造CS各层的参数向量,并初始化每个个体的卷积核和偏置参数
    void constructCS();
    //更新C3层,S2到C3的映射有点复杂,对不同的情况下可能会被重写.C3层有CS_NUM[2](=10)个特征图Map,
    //取5~14号卷积核与S2层相应Map进行卷积...下面的计算只适合S2有5个Map,C3有10个Map的情况
    void updateC3Layer(const kernelType* c_kernel,const biasType& bias);
	//通过全连接形式更新C3层,每一个C3层图对每一个S2层的图都需要一个卷积核
	void updateC3AllConnect(const kernelType* c_kernel,const biasType& bias);
    //更新C5层,C5层有CS_NUM[4](=20)个特征图Map,取0~20号卷积核分别与S4层每个Map进行卷积,C5层每个Map只有一个值
    void updateC5Layer(const kernelType* c_kernel,const biasType& bias);
    //更新S6层,S6层有CS_NUM[5](=11)个特征图Map,S6中每个输出的特征值从C5中随机取两个Map取偏置求和得到
    void updateS6Layer(const biasType& bias);
protected:	//静态保护方法
    //把out与vec1相加保存到output
    static void addVector(vectorF2D& output,const vectorF2D& vec1);
	template<typename...Args>
	static void addVector(vectorF2D &out,const vectorF2D &vec1,const Args...args){
		addVector(out,vec1);
		addVector(out,args...);
	}
    //对一个特征图input与卷积核kernel执行卷积操作再加偏置bias,得到对应的特征映射图output
    static void convoluteMap(vectorF2D& output,const vectorF2D& input,const kernelType& kernel,float _bias);
    //对一个特征图input通过mode方式下采样后再加偏置bias,得到对应的特征提取图output
    static void subSampleMap(vectorF2D& output,const vectorF2D& input,float _bias,int mode = SSM_Mean);
    //得到从vec[row][col]位置开始与卷积核kel卷积后的结果
    static float getConvoluteOut(const vectorF2D& vec, const kernelType& kernel, int row, int col)
    {
        uint i,j;
        float result = 0;
        for(i=row; i<kernelSize+row && i<vec.size(); ++i)
            for(j=col; j<kernelSize+col && j<vec[i].size(); ++j)
                result += vec[i][j]*kernel[i-row][j-col];
        return result;
    }
    //得到以vec[row][col]位置开始的亚采样结果
    static float getSubSampleOut(const vectorF2D& vec, int row, int col,int mode = SSM_Mean)
    {
        uint i,j;
        float result;
        if(mode==SSM_Min)
            result = FLT_MAX;
        else if(mode==SSM_Max)
            result = -FLT_MAX;
        else
            result = 0;
        for(i=row; i<subSampleSize+row && i<vec.size(); ++i)
            for(j=col; j<subSampleSize+col && j<vec[i].size(); ++j) {
                if(mode == SSM_Mean)
                    result += vec[i][j];
                else if(mode == SSM_Max) {
                    if(result<vec[i][j])
                        result = vec[i][j];
                } else if(mode == SSM_Min) {
                    if(result>vec[i][j])
                        result = vec[i][j];
                }
            }
        if(mode==SSM_Mean)
            result /= (subSampleSize*subSampleSize);
        return result;
    }
	//将一个多图加起来的Map图每个点求sigmoid
	static void sigmoidMap(vectorF2D &vec)
	{
		for(uint i=0;i<vec.size();++i)
			for(uint j=0;j<vec[i].size();++j)
				vec[i][j] = (float)sigmoid(vec[i][j]);
	}
	//将一个二维vector内数据全部清空,每个元素均变为0
	static void zeroVector(vectorF2D &vec)
	{
		for(uint i=0;i<vec.size();++i)
			for(uint j=0;j<vec[i].size();++j)
				vec[i][j] = 0;
	}
};

#ifndef _MSC_VER
#include"CNN_Base.cpp"
#endif

#endif
