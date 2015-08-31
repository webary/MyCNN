#include "Param.hpp"

int  Param::trainSetNum;	//训练集个数
int  Param::testSetNum;		//测试集个数
bool Param::testHaveTag;	//测试集数据中有无标签
char Param::trainFile[128];	//训练集文件
char Param::testFile[128];	//测试集文件
bool Param::enableTest;		//开启测试
float Param::F_CNN,Param::F_Matrix;	//卷积网络和矩阵网络的变异因子
int Param::MaxGen;		//最大演化代数
bool Param::multiThread;//开启多线程
bool Param::askLoadBak;		//询问载入备份文件
bool Param::loadBakFile;	//载入备份文件,不询问时使用
int Param::outputNums;				//输出模块数 
INI_Util Param::ini;
