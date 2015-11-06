/*
 * Copyright (c) 2015
 * 闻波, webary, HBUT 
 * reference ：https://github.com/webary/MyCNN/
 *             http://www.cnblogs.com/webary/
 */
 
#ifndef _MATH_UTIL_HPP_
#define _MATH_UTIL_HPP_

#define _CRT_SECURE_NO_WARNINGS
#include<cmath>
#include<ctime>
#include<cstdlib>
#include<string>
#include "Win_Util.h"
using namespace std;

/**
 * 数学操作类函数通用类
**/
class Math_Util {
public:
	/* 以下两个函数 非线性，单调，无限次可微
	 * |net|较小时（权值较小），可近似线性函数——高增益区处理小信号
	 * |net|较大时（权值较大），可近似阈值函数——低增益区处理大信号
	 */
	//sigmoid函数（S型函数，连续可微，值域(0，1)）
	static double sigmoid(double x) {
		return 1.0/(1.0+exp(-x));
	}
	//双曲正切S型函数，连续可微，值域(-1，1)）
	static double tansig(double x) {
		return 2 * sigmoid(x) - 1;
	}
	//仅设置一次随机数种子
	static void setSrand() {
		static bool first = 1;
		if(first) {
			srand(unsigned(time(NULL)));
			first = 0;
		}
	}
	//产生一个_min到_max之间的均匀分布的随机浮点数
	static float randFloat() {
		return (float)rand() / RAND_MAX;
	}
	//产生一个_min到_max之间的均匀分布的随机浮点数
	static float randFloat(double _min, double _max) {
		return (float)(randFloat() * (_max - _min) + _min);
	}
	//产生一个均指为E方差为D的高斯分布的随机浮点数
	static float randGauss(double E=0,double D=1) {
		float V1, V2=0, S=1, X;
		static bool phase = 0;
		if (phase == 0) {
			do {
				V1 = 2 * randFloat() - 1;
				V2 = 2 * randFloat() - 1;
				S = V1 * V1 + V2 * V2;
			} while (S >= 1 || S < 1e-5);
			X = V1 * sqrt(-2 * log(S) / S);
		} else
			X = V2 * sqrt(-2 * log(S) / S);
		phase = !phase;
		return float(X * D + E);
	}
	//获取当前时间,返回为字符串
	static char* getTime(char* timeStr) {
		time_t now_time=time(0);
		struct tm *newtime=0;
		LOCALTIME(newtime, &now_time);
		strftime(timeStr,10,"%H.%M.%S", newtime);
		return timeStr;
	}
	//获取当前日期和时间,返回为字符串
	static string getDateTime(bool noDate=0, char format=':') {
		char dt[20];
		time_t now_time=time(0);
		struct tm *newtime=0;
		LOCALTIME(newtime, &now_time);
		if(format=='.') {
			if(noDate)
				strftime(dt,20,"%H.%M.%S", newtime);
			else
				strftime(dt,20,"%Y-%m-%d %H.%M.%S", newtime);
		} else {
			if(noDate)
				strftime(dt,20,"%X", newtime);
			else
				strftime(dt,20,"%Y-%m-%d %X", newtime);
		}
		return dt;
	}
	//判等模板
	template<class T1,class T2>
	static bool myEqual(T1 a,T2 b) {
		return (a-b)<1e-5 && (b-a)<1e-5;
	}
	//使一个变量的值控制在某个范围内
	template<typename T1, typename T2, typename T3>
	static void makeInRange(T1& var, T2 low, T3 high, bool edgeToRand=true) {
		if(var<low)
			var = edgeToRand ? randFloat(low,high) : (T1)low;
		else if(var>high)
			var = edgeToRand ? randFloat(low,high) : (T1)high;
	}
	//返回绝对值
	template<typename T>
	static T myAbs(T num) {
		return num > 0 ? num : -num;
	}
	//转化模版————归一化函数
	template<class T1,class T2>
	static T1 trans(T2 num,T2 max_T2,T1 stand) {
		return  T1((double)num / max_T2 * stand + .5);
	}
};

#endif // _MATH_UTIL_HPP_
