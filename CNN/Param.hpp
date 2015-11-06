/*
 * Copyright (c) 2015
 * 闻波, webary, HBUT 
 * reference ：https://github.com/webary/MyCNN/
 *             http://www.cnblogs.com/webary/
 */
 
#pragma once
#ifndef  _Param_HPP_
#define  _Param_HPP_

#include "INI_Util.hpp"
#include "Win_Util.h"
#include <cstdlib>

#define _STR(s) #s<<" = "
#define _STR_STR(s) #s<<" = "<<s

/**
	参数读取类
**/
class Param{
public:
	static INI_Util ini;
	static int  trainSetNum;	//训练集个数
	static int  testSetNum;		//测试集个数
	static bool testHaveTag;	//测试集数据中有无标签
	static char trainFile[128];	//训练集文件
	static char testFile[128];	//测试集文件
	static bool enableTest;		//开启测试
	static float F_CNN, F_Matrix;//卷积网络和矩阵网络的变异因子
	static int 	MaxGen;         //最大演化代数
	static bool multiThread;	//开启多线程
	static bool askLoadBak;		//询问载入备份文件
	static bool loadBakFile;	//载入备份文件,不询问时使用
	static int outputNums;		//输出模块数 
public:
	//从ini文件载入配置参数
	static void loadINI(const char *iniFileName="",const char *appName=""){
		ini.loadINI(iniFileName);
		if(appName!=0 && appName[0]!=0)
			ini.setDefaultNode(appName);
		if(ini.isNodeExist(ini.getDefaultNode())==0) {
			writeINI();
			char msg[128];
			SPRINTF(msg,"未检测到正确的配置文件参数!\n\n请正确设置配置文件'%s'中的[%s]节点各参数"
					,iniFileName,appName);
			cout<<msg<<endl;
			throw "未检测到正确的配置文件参数!";
		}
		trainSetNum = getInt("trainSetNum");
		testSetNum = getInt("testSetNum");
		outputNums = getInt("outputNums"); 
		MaxGen = getInt("MaxGen");
		testHaveTag = 0!=getInt("testHaveTag");
		enableTest = 0!=getInt("enableTest");
		multiThread = 0!=getInt("multiThread");
		askLoadBak = 0!=getInt("askLoadBak");
		loadBakFile = 0!=getInt("loadBakFile");
		F_CNN = (float)getDouble("F_CNN");
		F_Matrix = (float)getDouble("F_Matrix");
		SPRINTF(testFile,"%s",ini.getRecord("testFile").c_str());
		SPRINTF(trainFile,"%s",ini.getRecord("trainFile").c_str());
	}
	//将参数保存到ini文件,当文件不存在时写入默认值
	static void writeINI() {
		ofstream write(ini.getLastFileName().c_str(),ios::out|ios::app);
		if(write.is_open()) {
			write<<"\n["<<ini.getDefaultNode()<<"]"<<endl
			<<_STR(trainFile)<<endl
			<<_STR(testFile)<<endl
			<<_STR(trainSetNum)<<endl
			<<_STR(testSetNum)<<endl
			<<_STR(outputNums)<<endl
			<<_STR(testHaveTag)<<1<<endl
			<<_STR(enableTest)<<0<<endl
			<<_STR(F_CNN)<<0.07<<endl
			<<_STR(F_Matrix)<<0.08<<endl
			<<_STR(MaxGen)<<100000000<<endl
			<<_STR(multiThread)<<1<<endl
			<<_STR(askLoadBak)<<1<<endl
			<<_STR(loadBakFile)<<1<<endl;
			write.close();
		}
	}

	static int getInt(const string & str) {
		string res = ini.getRecord(str);
		if(res!="")
			return atoi(res.c_str());
		else
			return 0;
	}
	static double getDouble(const string & str) {
		string res = ini.getRecord(str);
		if(res!="")
			return atof(res.c_str());
		else
			return 0;
	}
	static string getState() {
		return ini.getState();
	}
};

#ifndef _MSC_VER
#include"Param.cpp"
#endif

#endif // _INI_Util_HPP_
