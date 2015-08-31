#ifndef _ADAMACHINELEARNING_HPP_
#define _CRT_SECURE_NO_WARNINGS
#include "adaMachineLearning_w.hpp"
//#include "stdafx.h"
#endif

#ifdef __AFXWIN_H__
int testOut[1000];
char ml_info[10000];
#endif

int flag_stop = -1;

Ada_ML::Ada_ML(int MLDimension,bool _mustEqual,int _col)
{
	setSrand();
	PEout = new int*[popSize];
	dataLen = 0;
	outPutModels = 1;	//默认只有一个输出模块,即输出0或1
	setML(MLDimension, _mustEqual, _col);
	memset(&cBest,0,sizeof(cBest));
	b_saveTmp = b_savePre = b_saveBest = 0;
	wait = 5000;
#ifdef __AFXWIN_H__
	pS = 0;
#endif
}

Ada_ML::~Ada_ML()
{
	delete[] PEout;
}

inline void Ada_ML::setML(int MLDimension,bool _mustEqual,int _col)
{
	mlDimension = MLDimension;
	//(d+23)/24*8 ==> [1,24]=>8, [25,48]=>16, [49,72]=>24, [73,96]=>32...
	numPerCol = mlDimension;
	//(mlDimension + OP_NUMSx3 - 1) / OP_NUMSx3 * OP_NUMS;
	col = _col;
	chromLength = col * 3 * numPerCol + 3 * outPutModels;
	b_mustEqual = _mustEqual;
}

void Ada_ML::setWaitCol(double wt,int _col)
{
	wait = wt*1000;
	col = _col?_col:col;
	chromLength = col * 3 * numPerCol + 3 * outPutModels;
}

void Ada_ML::setOutputModels(int outModels)
{
	outPutModels = outModels;
	chromLength = col * 3 * numPerCol + 3 * outPutModels;
}

inline void Ada_ML::setSave(bool saveBest, bool saveTmp, bool savePre)
{
	b_saveBest = saveBest, b_saveTmp = saveTmp, b_savePre = savePre;
}
//将矩阵中个体indiv第i维变为随机范围内的值
inline int Ada_ML::getChromRandValue(int i)
{
	if((i-2)%3==0)	//每到第三个数变为随机操作符的索引
		return rand() % OP_NUMS;
	else {
		if(i<3 * numPerCol) //矩阵的第一列从前面mlDimension个神经元的输出值中取
			return rand() % mlDimension;
		else	//矩阵的后面列从前一列取
			return rand() % numPerCol;
	}
}

void Ada_ML::init(cchar *readFromFile,bool b_saveData)
{
	// readFromFile指向读入一条预存路径的文件
	// b_saveData记录是否保存需要处理的数据集
	int i, j, k;
	static int first = 1;
	for(i=0; i<(first?popSize:popSize-1); i++) {
		memset(&pop[i],0,sizeof(pop[i]));
		if(i==0 && readFromFile && readFromFile[0]) {
			if(loadBest(readFromFile)) {
				pop[i] = cBest;
			} else {		//如果文件没找到,则不从文件中得读取
				readFromFile = 0;
				i--;
			}
		} else {
			for (j=0; j<chromLength; j++)
				pop[i].chrom[j] = getChromRandValue(j);
			for(j=0; j<numPerCol; j++) {
				for(k=0; k<mlDimension; k++)
					pop[i].ww[j][k] = randFloat(-1,1);
				pop[i].ww[j][mlDimension] = randFloat(-.5,.5);
			}
		}
	}
	bool b_saveLoad = 0;//标记是否保存载入进来的路径文件数据，来校对载入是否正确
	if(b_saveLoad) {
		ofstream saveBest;
		saveBest.open("save.pth");
		for(i = 0; i < chromLength; i++)
			saveBest<<(int)cBest.chrom[i]<<" ";
		saveBest<<endl;
		for(j=0; j<numPerCol; j++)
			for(k=0; k<mlDimension+1; k++)
				saveBest<<setw(12)<<setiosflags(ios::left)<<cBest.ww[j][k]<<" ";
		saveBest.close();
	}
	//在后面的演化中把最后一个个体改为前一次演化后的最好值
	if(first)
		first = 0;
	else
		pop[popSize-1] = cBest;
	if(b_saveData) {
		ofstream fout("dataSave.txt");
		if(fout.is_open()) {
			for(i = 0; i < dataLen; i++) {
				for(j = 0; j < mlDimension; j++)
					fout<<matTrain[i].data[j]<<"\t";
				fout<<matTrain[i].data[j]<<endl;
			}
			fout.close();
		}
	}
}

inline int Ada_ML::operationOnXY(int x,int y,int op)
{
	switch (op%OP_NUMS) {
	case 0:
		return x&y;
		break;
	case 1:
		return x|y;
		break;
	case 2:
		return !x;
		break;
	case 3:
		return !y;
		break;
	case 4:
		return x==y;
		break;
	case 5:
		return x!=y;
		break;
	case 6:
		return x;
		break;
	case 7:
		return y;
		break;
	default:
		return 0;
	}
}

inline int Ada_ML::getPEOut(Individual& indiv, const vectorF &dataRow)
{
	getNNOut(indiv,dataRow);
	return getMatOut(indiv);
}

int Ada_ML::getMatOut(const Individual& indiv)
{
	int x, y, op, row = 0, _col = 0, j = 0;
	for(j = 0; j < chromLength - 3 * outPutModels; j+=3) {
		if(j%(3*numPerCol)==0) {	//一列处理完后进入下一列
			row = 0;
			_col = j / (3*numPerCol);
		}
		if(_col==0) {	//才进入模版时从神经网络输出中取数据
			x = indiv.wwOut[indiv.chrom[j]];
			y = indiv.wwOut[indiv.chrom[j+1]];
		} else {	//从前一列取数据填充当前列
			x = PE[indiv.chrom[j]][_col-1];
			y = PE[indiv.chrom[j+1]][_col-1];
		}
		op = indiv.chrom[j+2];
		PE[row++][_col] = operationOnXY(x,y,op);
	}
	int matOutput = 0;
	for( ; j < chromLength ; j+=3 ) {
		x  = PE[indiv.chrom[j]][_col-1];
		y  = PE[indiv.chrom[j+1]][_col-1];
		op = indiv.chrom[j+2];
		matOutput = 2 * matOutput + operationOnXY(x,y,op);
	}
	return matOutput;
}

inline int Ada_ML::getBestPEOut(const vectorF &dataRow)
{
	return getPEOut(cBest,dataRow);
}

void Ada_ML::getNNOut(Individual& indiv, const vectorF &dataRow)
{
	for(int i=0; i<mlDimension; ++i) {
		// double sum = 0;
		// for(int j=0; j<mlDimension; ++j)
		// sum += dataRow[j] * indiv.ww[i][j];
		// indiv.wwOut[i] = tansig(sum) > indiv.ww[i][mlDimension];
		//2015.8.28 00:18 改为只有演化阈值的情况
		indiv.wwOut[i] = dataRow[i] > indiv.ww[i][mlDimension];
	}
}

void Ada_ML::fitnessValue(Individual& indiv, int ii)  	// ii < popSize
{
	indiv.fitness = 0;
	for(int row = 0; row < dataLen; row++) {
		PEout[ii][row] = getPEOut(indiv, matTrain[row].data);
		indiv.fitness += PEout[ii][row] != matTrain[row].tag;
	}
}

int  Ada_ML::findBest()
{
	int tag_best = 0, i;
	for(i = 1; i < popSize; i++)
		if(pop[i].fitness < pop[tag_best].fitness)
			tag_best = i;
	best = tag_best;
	cBest= pop[best];
	return best;
}
//高斯和多点方式变异个体的矩阵部分和神经网络部分
void Ada_ML::mutateMatrixNNByGauss(Individual& tmpMatPop)
{
	int i,j,k;
	//对矩阵左边的神经元参数执行高斯变异
	for (i = 0; i < (mlDimension+1)*mlDimension; i++)
		if (randFloat() <= F) {
			j = i/(mlDimension+1), k = i%(mlDimension+1);
			tmpMatPop.ww[j][k] += randGauss(0, .8);
			if(k==mlDimension)  //控制偏置参数的范围
				makeInRange(tmpMatPop.ww[j][k],-.5,.5);
			else  				//控制权值参数的范围
				makeInRange(tmpMatPop.ww[j][k],-1,1);
		}
	//对矩阵部分执行多点变异
	for (i=0; i<chromLength; i++)
		if(randFloat()<=F)
			tmpMatPop.chrom[i] ^= getChromRandValue(i);
}
//在变异前产生几个随机个体用于交叉步骤
inline void Ada_ML::generatePopRand(int ii)
{
	if(ii>=popSize) return;
	popRand[0] = rand() % popSize;
	popRand[1] = rand() % popSize;
	while(popRand[0]==ii)
		popRand[0] = rand() % popSize;
	while(popRand[1]==popRand[0]||popRand[1]==ii)
		popRand[1] = rand()%popSize;
}

void Ada_ML::mutateMatrixNNByDE(Individual& tmp,int ii,int *_popRand)  //最少需要3个个体
{
	if(ii>=popSize) return;
	int dis=0,j,j1,j2;
	int r1,r2;
	if(_popRand==0) {
		r1=rand() % popSize, r2=rand() % popSize;
		while(r1==ii)
			r1=rand() % popSize;
		while(r1==r2||r2==ii)
			r2=rand()%popSize;
	} else { //必须在执行此函数前执行generatePopRand
		r1 = popRand[0];
		r2 = popRand[1];
	}
	//变异矩阵模板部分
	for (j=0; j<chromLength; j++)
		if (pop[r1].chrom[j]!=pop[r2].chrom[j])
			dis++;
	int s = int(floor(F * dis));
	int flag1[MaxChromLength];
	for (j=0; j<s; j++)
		flag1[j] = rand()%chromLength;
	for(j2=0; j2<s; j2++)		//判重
		for (j1=0; j1<s; j1++)
			if(flag1[j2]==flag1[j1] && j2!=j1) {
				flag1[j2]=rand()%chromLength;
				j1=-1;
			}
	for (j1=0; j1<s; j1++)
		if (randFloat() <= CR)
			tmp.chrom[flag1[j1]] = getChromRandValue(flag1[j1]);
	//变异神经网络部分
	//int z = rand() % ((mlDimension+1)*mlDimension), n, m;
	//for (j = 0; j < (mlDimension+1)*mlDimension; j++)
	//    if (j == z || randFloat() <= CR) {
	//        n = j/(mlDimension+1), m = j%(mlDimension+1);
	//        tmp.ww[n][m] += F * (pop[best].ww[n][m] - pop[ii].ww[n][m] + pop[r1].ww[n][m] - pop[r2].ww[n][m]);
	//		if(m==mlDimension)  //控制偏置参数的范围
	//			makeInRange(tmp.ww[n][m],-.5,.5);
	//		else  				//控制权值参数的范围
	//			makeInRange(tmp.ww[n][m],-1,1);
	//    }
	///2015.8.28 改为只有演化阈值的情况
	int z = rand() % mlDimension, m = mlDimension;
	for (j = 0; j < mlDimension; j++)
		if (j == z || randFloat() <= CR) {
			tmp.ww[j][m] += F * (pop[best].ww[j][m] - pop[ii].ww[j][m] + pop[r1].ww[j][m] - pop[r2].ww[j][m]);
			makeInRange(tmp.ww[j][m],-.5,.5);  //控制偏置参数的范围
		}
}

inline void Ada_ML::mutate(int ii,int *_popRand)
{
	if(ii>=popSize) return;
	Individual tmp = pop[ii];//即将变异的临时个体
	//mutateMatrixNNByDE(tmp,ii,_popRand);
	mutateMatrixNNByGauss(tmp);
	selectPop(tmp,ii);
}
//
void Ada_ML::selectPop(Individual& indiv,int ii)
{
	int *tmpPEOut = new int[dataLen];
	memcpy(tmpPEOut,PEout[ii],dataLen*sizeof(int));
	fitnessValue(indiv,ii);
	if (indiv.fitness < pop[ii].fitness) {
		//cout<<"变异成功"<<endl;
		pop[ii]=indiv;
	} else {
		memcpy(PEout[ii],tmpPEOut,dataLen*sizeof(int));
		F  = randFloat();
		CR = randFloat();
	}
	delete[] tmpPEOut;
}

//创建工作者线程进行后台计算和判断
inline RETURN_TYPE waiting(void* tt)
{
	if(flag_stop==1)
		return RETURN_VALUE;
	flag_stop = 0;
	if(MessageBox(HAND,_T("此次演化时间稍长，是否现在停止演化？"), _T("温馨提示"),MB_YESNO|MB_ICONQUESTION)==IDYES)
		flag_stop = 1;
	else {
		flag_stop = -1;
		*(int*)tt = clock();
	}
	return RETURN_VALUE;
}

void Ada_ML::startLearn(int len,double permit_error,cchar *readFromFile,bool b_saveData)
{
	if(len>0 && len < dataLen)
		dataLen = len;
	if(dataLen==0) {
#ifndef  __AFXWIN_H__
		cout<<"\r数据集（训练集）个数设置不正确，请检查从文件载入是否正常！"<<endl;
		return;
#else
		MessageBox(AfxGetMainWnd()->m_hWnd, _T("数据集（训练集）个数设置不正确，请检查从文件载入是否正常！"), _T("模板演化出错"),0);
		if(pS)
			pS->SetPaneText(0,_T("模板演化出错--错误代码为0001"));
		return;
#endif
	}
	permitError = permit_error>0 ? permit_error : 8;
	int i, gen = 0;
	for(i = 0; i < popSize; i++)
		PEout[i] = new int[dataLen];
	ofstream saveTmp;
	if(b_saveTmp) { //先清空文件内容
		saveTmp.open("tmp.txt");
		saveTmp.close();
	}
	init(readFromFile,b_saveData);
	for(i = 0; i < popSize; i++)
		fitnessValue(pop[i],i);
	findBest();
	F = randFloat(),CR = randFloat();
	double ind_dif=100;
#ifndef  __AFXWIN_H__
	cout<<"\r正在处理"<<dataLen<<"组向量,并进行模板演化以查找最好路径..."<<endl;
#else
	if(pS)
		pS->SetPaneText(0,_T("已开始演化过程，请耐心等待..."));
#endif
	int t1 = clock(),tt1 = t1;
	flag_stop = -1;
	while (gen++<MaxGen) {
		for (i=0; i<popSize; i++)
			mutate(i);
		findBest();
		if(clock()-tt1>100 || gen==1) {
			tt1 = clock();
			if(b_mustEqual)
				ind_dif = cBest.fitness / dataLen * 100;
			/* else
			ind_dif = pow(cBest.fitness / dataLen,1.0 / Exp) / 2.55;*/
			ind_dif = int(ind_dif*100)/100.0;
#ifndef  __AFXWIN_H__
			cout<<"\r\t\t\t\t\t\r已演化至第 "<<gen<<" 代，平均误差为："<<ind_dif<<"     \b\b\b\b%";
#else
			SPRINTF(ml_info,"已演化至第 %d 代，平均误差为：%.3g %%",gen,ind_dif);
			if(pS)
				pS->SetPaneText(0,_T(ml_info));
			Sleep(1);
			MSG  msg;
			if (PeekMessage(&msg,(HWND)NULL, 0, 0, PM_REMOVE))
				::SendMessage(msg.hwnd, msg.message, msg.wParam, msg.lParam);
#endif
			static double last_ind_dif = 0;
			if(!myEqual(last_ind_dif,ind_dif)) {
				if(b_savePre)
					savePopMatrix("present.pth",best);
				if(b_saveTmp) {
					saveTmp.open("tmp.txt",ios::app);
					if(saveTmp.is_open()) {
						for(i = 0; i < dataLen; i++)
							saveTmp<<setw(2)<<(PEout[best][i]-matTrain[i].tag)<<" ";
						saveTmp<<"best = "<<best<<"\tind_dif = "<<ind_dif<<"%"<<endl;
						saveTmp.close();
					}
				}
				last_ind_dif = ind_dif;
				if(ind_dif < permitError+1e-5)
					break;
			}
			//flag_stop标识用户点击了对话框的哪个值，1表示确定，-1表示取消
			if(clock()-t1>wait && flag_stop==-1)
#ifdef  __AFXWIN_H__
				AfxBeginThread(waiting,&t1);	//MFC专用线程函数
#else
				_beginthread(waiting,0,&t1);
#endif
			if(flag_stop==1)
				break;
		}
		if(gen%1000==0)
			Sleep(1);
	}
	flag_stop=1;
#ifdef  __AFXWIN_H__
	SPRINTF(ml_info,"演化完成 (已演化至第 %d 代，平均误差为：%.3g %%)",gen,ind_dif);
	if(pS)
		pS->SetPaneText(0,_T(ml_info));
#else
	cout<<endl;
#endif
#ifndef  __AFXWIN_H__
	unsigned totalLen = matTrain.size();
	if (dataLen < totalLen) {
		cout << endl;
		double Terror = 0,error;
		for(i = dataLen; i < totalLen; i++) {
			if(i==dataLen) {
				cout<<"训练集个数: "<<dataLen<<"\t测试集个数: "
				    <<totalLen-dataLen<<endl
				    <<"模版输出结果"<<"\t数据对比结果"<<"\t对比误差\n";
			}
			int out = getPEOut(cBest,matTrain[i].data);
			cout<<"    "<<out<<"\t\t     "<<matTrain[i].tag;
			error = out-matTrain[i].tag;
			Terror += pow(error,2);
			cout<<"\t   "<<error<<endl;
		}
		Terror = (Terror/(totalLen-dataLen));
		cout<<"平均误差: "<<Terror<<endl;
	}
#endif
	if(b_saveBest)
		savePopMatrix("best.pth",best);
	for(i = 0; i < popSize; i++)
		delete[] PEout[i];
}

#ifdef __AFXWIN_H__
void Ada_ML::startLearn(int len,CStatusBar *p,double permit_error,cchar * readFromFile,bool b_saveData)
{
	pS = p;
	startLearn(len,permit_error,balance_r,readFromFile,b_saveData);
}
#endif
//从文件载入输入数据
int Ada_ML::loadInputData(const string &fileName,int size,vector<S_Input> &dataVec,bool haveTag)
{
	ifstream loadFile(fileName.c_str());
	if(loadFile.is_open()) {
		dataVec.clear();
		if(size>0)
			dataVec.reserve(size);
		uint i,j;
		bool success = true;	//如果某个数据读取错误，则失败
		S_Input input = {vectorF(mlDimension),0};
		for(i = 0; size < 1 || i < size; ++i) {
			for(j = 0; j < mlDimension && success; ++j)
				if(!(loadFile>>input.data[j])) {
					success = false;
					break;
				}
			//该组数据没有读取完整，则不加入训练集
			if(!success || haveTag && !(loadFile>>input.tag))
				break;
			dataVec.push_back(input);
		}
		loadFile.close();
		return i;
	} else
		return 0;
}
//从文件载入训练数据
int Ada_ML::loadTrainSet(const string &fileName,int size)
{
	if(fileName=="") {
		cout<<"\r请正确设置训练集文件路径"<<endl;
		throw "请正确设置训练集文件路径";
	}
	dataLen = loadInputData(fileName,size,matTrain,1);
	if(dataLen>0)
		cout<<"\r\t\t\t\t\t\r>>成功载入"<< dataLen <<"组训练集"<<endl;
	else
		cout<<"\r\t\t\t\t\t\r文件‘"<<fileName<<"’打开失败！"<<endl;
	return dataLen;
}

int Ada_ML::loadTestSet(const string &fileName, int size, bool haveTag)
{
	if(fileName=="") {
		cout<<"\r请正确设置测试集文件路径"<<endl;
		throw "请正确设置测试集文件路径";
	}
	int i = loadInputData(fileName,size,matTest,haveTag);
	if(i>0)
		cout<<"\r\t\t\t\t\t\r>>成功载入"<<i<<"组测试集"<<endl;
	else
		cout<<"\r\t\t\t\t\t\r文件‘"<<fileName<<"’打开失败！"<<endl;
	return i;
}
//文件名，数据组数，打印0或1的比例
double Ada_ML::getTestOut(const string &fileName, int len,int print01Ratio)
{
	int testLen = loadTestSet(fileName, len), res;
	//CString s;s.Format(_T("%d"),testLen);
	//MessageBox(0,s,"测试集长度",0);
	//for(int j=0;j<mlDimension;++j)
	//	cout<<matTest[0].data[j]<<" ";
	int pOut01[2] = {0};
	bool print01 = (print01Ratio==0 || print01Ratio==1);
	if(print01)
		pOut01[0] = pOut01[1] = 0;
	for(int i = 0; i < testLen; i++) {
		res = getPEOut(cBest,matTest[i].data);
		if(print01 && (res==0 || res==1))
			++pOut01[res];
#ifdef __AFXWIN_H__
		testOut[i] = res;
#else
		if(i==0)
			cout<<testLen<<"组测试数据模版输出:"<<endl<<res<<" ";
		else
			cout<<res<<" ";
#endif
	}
#ifndef __AFXWIN_H__
	cout<<endl;
	if(print01)
		cout<<"——————<0>\t<1>\t<total>\t<"<<print01Ratio<<">/<total>: \n"
		    <<"——————"<<pOut01[0]<<"\t"<<pOut01[1]<<"\t"<<pOut01[0]+pOut01[1]
		    <<"\t"<<setprecision(3)<<double(pOut01[print01Ratio])/(pOut01[0]+pOut01[1])<<endl<<endl;
#endif
	if(!print01) print01Ratio = 0;
	return double(pOut01[print01Ratio])/(pOut01[0]+pOut01[1]);
}

//读入数据并存入到最好个体中
int  Ada_ML::loadBest(cchar *filePath)
{
	if(filePath==0)
		return 0;
	int i,j,read;
	ifstream fin(filePath);
	if(fin.is_open()) {
		for(i=0; i<chromLength; ++i)
			if(fin>>read)
				cBest.chrom[i] = read;
			else
				return 0;
		for(i=0; i<numPerCol; ++i)
			for(j=0; j<mlDimension+1; ++j)
				if(!(fin>>cBest.ww[i][j]))
					return 0;
		fin.close();
		return 1;
	}
	return 0;
}

void Ada_ML::savePopMatrix(cchar* file,int index,ios::openmode mode)
{
	if(file==0 || index<0 || index>=popSize)
		return ;
	ofstream savePop(file,mode);
	if(savePop.is_open()) {
		int i,j;
		for(i = 0; i < chromLength; i++)
			savePop<<(int)pop[index].chrom[i]<<" ";
		savePop<<endl;
		for(i=0; i<mlDimension; i++) {
			for(j=0; j<mlDimension; j++)
				savePop<<setw(12)<<setiosflags(ios::left)<<pop[index].ww[i][j]<<" ";
			savePop<<pop[index].ww[i][j]<<endl;
		}
		if(!(mode & ios::app))	//非追加模式才输出这些信息
			savePop<<endl<<pop[index].fitness/dataLen<<endl;
		savePop.close();
	}
}

/**用法:
	//test1:
	Ada_ML ml(15,0);
	ml.loadDataSet("train.txt");
	//ml.setSave(0,1);
	ml.startLearn(30,5);
	//ml.getTestOut("test.txt");

	//test2:
    Ada_ML ml2(4,1,1);
    ml2.loadDataSet("t.txt",-1,1,1);
    ml2.setWait(4);
    ml2.startLearn(14,1);
    ml2.getTestOut("tt.txt",-1,1);
**/
