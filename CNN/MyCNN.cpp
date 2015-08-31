#include <direct.h>
#include "MyCNN.hpp"
#include "multiThread.h"

double MyCNN::F_CNN, MyCNN::F_Matrix;//在startCNNLearn成员函数中从配置文件获取初始化参数

MyCNN::MyCNN(uint modelNums,uint _inputSize,bool _mustEqual): Ada_ML(CS_NUM[4],_mustEqual),CNN_Base(_inputSize,popSize)
{
    bestPopIndex = 0;
    setWaitCol(10);	//设置默认询问等待间隔时间
    setOutputModels(modelNums);
}

MyCNN::~MyCNN() {}
//开始卷积网络学习，把普通神经网络的学习作为其中一部分
void MyCNN::startCNNLearn(double permit_error,cchar *readFromFile)
{
    F_CNN = Param::F_CNN;
    F_Matrix = Param::F_Matrix;
    if(cnnTrain.empty()) {
#ifndef  __AFXWIN_H__	//非MFC环境
        cout<<"\r尚未载入输入图数据,请检查从是否已正常载入！"<<endl;
        throw "无训练集数据,从startCNNLearn中退出！";
        return;
#else		//MFC环境
        MessageBox(AfxGetMainWnd()->m_hWnd,"尚未载入输入图数据,请检查从是否已正常载入！","卷积学习出错",0);
        if(pS)
            pS->SetPaneText(0,_T("卷积学习出错--错误代码0x0001"));
        return;
#endif
    }
    setBlue();
    permitError = (float)(permit_error>0 && permit_error<100 ? permit_error : 8);
    int i, gen = 0;
    matTrain.resize(cnnTrain.size(),{vectorF(mlDimension)});
    if(!cnnTest.empty())
        matTest.resize(cnnTest.size(),{vectorF(mlDimension)});
    for(i = 0; i < popSize; i++)
        PEout[i] = new int[dataLen];
    cout<<"\n正在初始化神经网络...";
    init();	//初始化神经网络部分
    loadBestPop(readFromFile);
    for(i = 0; i < popSize; ++i)
        getFitValue(cnnPop[i],pop[i]);	//计算每个个体的适应值
    findBest();
#ifndef  __AFXWIN_H__
    cout<<"\r正在学习"<<cnnTrain.size()<<"组训练集,并训练卷积神经网络..."<<endl;
#else
    if(pS)
        pS->SetPaneText(0,_T("已开始演化过程，请耐心等待..."));
#endif
    setWhite();
    cout<<"\n当前代数:\t平均正确率:\t此代耗时(ms):\t当前测试正确率:\t历史测试正确率:"<<endl;
    flag_stop = -1;
    int t1 = clock(), t_start = t1;
    double ind_right = 100;		//平均误差
    static double last_ind_right = 0;	//上一次的平均正确率
    float bestTestRight = 0;	//历史最好测试正确率
    _mkdir("Gen_TestRight");	//创建文件夹
    char tmpbuf[128];
    SPRINTF(tmpbuf,"Gen_TestRight/Gen_TestRight_%s.txt",getDateTime(0,'.').c_str());
    ofstream saveGen_TestRight(tmpbuf);
    saveGen_TestRight<<"训练集文件及个数："<<Param::trainFile<<"  "<<Param::trainSetNum<<endl;
    saveGen_TestRight<<"测试集文件及个数："<<Param::testFile<<"  "<<Param::testSetNum<<endl;
    saveGen_TestRight<<"F_CNN = "<<F_CNN<<",F_Matrix = "<<F_Matrix<<endl<<endl
                     <<"代数: \t\t平均正确率: \t测试正确率: "<<endl;
    while (gen++<Param::MaxGen) {
        int size = popSize;
        //添加子线程辅助计算一半个体
        if(Param::multiThread) {
            MyCNN_Index cnn_index = {this,Mutate_Half};
            _beginthread(MyCNN_Thread,0,&cnn_index);
            size /= 2;
        }
        for (i=0; i<size; i++)
            mutate(i); //对个体执行变异
        if(Param::multiThread)
            waitForFinish();
        findBest();
        ind_right = 1-(cnnPop[bestPopIndex].fitValue) / cnnTrain.size();
        cout<<"\r "<<setw(17)<<setiosflags(ios::left)<<gen;
        setGreen();
        cout<<setw(16)<<ind_right;
        setWhite();
        cout<<setw(17)<<clock()-t_start;
        if(ind_right*100 > 100-permitError+1e-5)
            break;
        //flag_stop标识用户点击了对话框的哪个值，1表示确定，-1表示取消
        if(clock()-t1>wait && flag_stop==-1)
#ifdef  __AFXWIN_H__
            AfxBeginThread(waiting,&t1);	//MFC专用线程函数
#else
            _beginthread(waiting,0,&t1);
#endif
        if(flag_stop==1)
            break;
        if(!myEqual(last_ind_right,ind_right)) {
            float testRight = 0;
            //从配置文件读取是否要进行测试
            Param::loadINI();
            if(Param::enableTest) {
				testRight = compareTestOut();	//当前最好个体对测试集的正确率
				setGreen();
				cout << setw(17) << testRight;
        		setWhite();
				if (bestTestRight < testRight)
					bestTestRight = testRight;
				if (testRight < bestTestRight)
					cout << bestTestRight;
				else
					cout << "==";
            }
            cout<<endl;	//加换行可以让每次变化后都显示前一次的代数
            if(saveGen_TestRight.is_open())
                saveGen_TestRight<<gen<<"\t\t"<<setw(16)<<setiosflags(ios::left)<<ind_right<<testRight<<endl;
            //保存当前最好个体的可演化参数到文件-用来备份
            saveBestToFile("BestPop.bak");
            last_ind_right = ind_right;
        }
        t_start = clock();
    }
    if(saveGen_TestRight.is_open())
        saveGen_TestRight.close();
    flag_stop=1;
#ifdef  __AFXWIN_H__
    SPRINTF(ml_info,"演化完成 (已演化至第 %d 代，平均正确率为：%.3g)",gen,ind_right);
    if(pS)
        pS->SetPaneText(0,_T(ml_info));
#else
    cout<<endl;
#endif
    for(i = 0; i < popSize; i++)
        delete[] PEout[i];
}
//得到测试集的输出
void MyCNN::getTestOut()
{

}
//对比测试集的输出正确率---前提是已知每组测试数据的输出
float MyCNN::compareTestOut()
{
    int error = 0;	//记录比较后出错的个数
    uint i_test;
    for(i_test=0; i_test<cnnTest.size(); ++i_test) {
        updatePerLayer(cnnTest[i_test].data,cnnPop[bestPopIndex]);
        getCNNOut(matTest[i_test].data);
        error += (int)(cnnTest[i_test].tag != getPEOut(pop[bestPopIndex], matTest[i_test].data));
    }
    if(i_test==0) {
        cout<<"还未载入测试集,无法完成识别！"<<endl;
        return -1;
    } else {
        float right = 1 - (float)error/i_test;
        //cout<<"对测试数据的识别正确率为: "<<right<<"("<<error<<"/"<<i_test<<")"<<endl;
        return right;
    }
}
//保存最好个体的卷积网络可训练参数
void MyCNN::saveBestCnnPop(const char* file)
{
    ofstream outFile(file);
    uint index = bestPopIndex, i, j ,k;
    for(i=0; i<kernelNum; ++i) {
        outFile<<"cnnPop-best"<<"  kernel-"<<i<<endl;
        for(j=0; j<cnnPop[index].c_kernel[i].size(); ++j) {
            for(k=0; k<cnnPop[index].c_kernel[i][j].size(); ++k)
                outFile<<cnnPop[index].c_kernel[i][j][k]<<"  ";
            outFile<<endl;
        }
        outFile<<"————————————————————————————"<<endl;
    }
    outFile<<"bias-"<<index<<endl;
    for(i=0; i<cnnPop[index].bias.size(); ++i)
        outFile<<cnnPop[index].bias[i]<<"  ";
    outFile<<endl;
    outFile.close();
}
//保存最好个体的矩阵网络参数
void MyCNN::saveBestMatPop(const char* file)
{
    ofstream savePre(file);
    if(savePre.is_open()) {
        int i,j,k;
        for(i = 0; i < chromLength; i++)
            savePre<<(int)cBest.chrom[i]<<" ";
        savePre<<endl;
        for(j=0; j<mlDimension; j++)
            for(k=0; k<mlDimension+1; k++)
                savePre<<setw(12)<<setiosflags(ios::left)<<cBest.ww[j][k]<<" ";
        savePre<<endl;
        savePre.close();
    }
}
//保存当前种群最好个体的可演化参数到文件中
void MyCNN::saveBestToFile(const char* file)
{
    ofstream outFile(file);
    if(outFile.is_open()) {
        outFile<<"### 备份于: "<<getDateTime()<<"\t正确率为: "
               <<1-(cnnPop[bestPopIndex].fitValue) / cnnTrain.size()<<endl<<endl;
        uint i,j,k;
        for(i=0; i<kernelNum; ++i) {
            for(j=0; j<cnnPop[bestPopIndex].c_kernel[i].size(); ++j) {
                for(k=0; k<cnnPop[bestPopIndex].c_kernel[i][j].size(); ++k)
                    outFile<<cnnPop[bestPopIndex].c_kernel[i][j][k]<<"  ";
                outFile<<endl;
            }
            outFile<<endl;
        }
        outFile<<endl;
        for(i=0; i<cnnPop[bestPopIndex].bias.size(); ++i)
            outFile<<cnnPop[bestPopIndex].bias[i]<<"  ";
        outFile<<"\n\n"<<cnnPop[bestPopIndex].stepInGuass[0]
               <<"\t"<<cnnPop[bestPopIndex].stepInGuass[1]<<endl<<endl;
        outFile.close();
        //输出矩阵部分到该文件
        savePopMatrix(file,bestPopIndex,ios::out|ios::app);
        //cout<<"已输出到文件"<<file<<"！"<<endl;
    }
}
//从文件载入备份的数据
void MyCNN::loadBestPop(cchar *filePath)
{
    ifstream bestPopFile(filePath);
    if(bestPopFile.is_open()) {
        if(MessageBox(0,_T("检测到有备份文件可载入,是否载入？"), _T("温馨提示"),MB_ICONQUESTION|MB_YESNO)==IDYES) {
            string tmp = "";
            getline(bestPopFile,tmp);	//读取第一行注释，丢弃
            uint i,j,k;
            for(i=0; i<kernelNum; ++i) {
                for(j=0; j<cnnPop[0].c_kernel[i].size(); ++j) {
                    for(k=0; k<cnnPop[0].c_kernel[i][j].size(); ++k)
                        bestPopFile>>cnnPop[0].c_kernel[i][j][k];
                }
            }
            for(i=0; i<cnnPop[0].bias.size(); ++i)
                bestPopFile>>cnnPop[0].bias[i];
            bestPopFile>>cnnPop[0].stepInGuass[0]>>cnnPop[0].stepInGuass[1];
            //读入矩阵部分的参数
            int read;
            for(i=0; i<chromLength; ++i)
                if(bestPopFile>>read)
                    pop[0].chrom[i] = read;
                else
                    break;
            for(i=0; i<mlDimension; ++i)
                for(j=0; j<mlDimension+1; ++j)
                    if(!(bestPopFile>>pop[0].ww[i][j]))
                        break;
        }
        bestPopFile.close();
    }
}
//对整个个体执行变异,包括卷积网络部分和矩阵网络部分
void MyCNN::mutate(uint index)
{
    CNNIndividual tmpCnnPop;
    Individual tmpMatPop;
    tmpCnnPop = cnnPop[index];
    tmpMatPop = pop[index];
    mutateByGauss_MultiPoint(tmpCnnPop,tmpMatPop);
    getFitValue(tmpCnnPop,tmpMatPop);	//得到变异个体的适应值
    if(tmpCnnPop.fitValue < cnnPop[index].fitValue) {
        cnnPop[index] = tmpCnnPop;
        pop[index] = tmpMatPop;
    }
}
//使用高斯变异和多点变异构造新新个体---mutate的一种实现方式
void MyCNN::mutateByGauss_MultiPoint(CNNIndividual& tmpCnnPop, Individual& tmpMatPop)
{
    static const float gaussRange = 1.5;    //高斯变异的方差
    uint i,j,k;
    //对卷积核执行高斯变异
    for(i=0; i<kernelNum; ++i)
        for(j=0; j<kernelSize; ++j)
            for(k=0; k<kernelSize; ++k)
                if(randFloat()<=F_CNN) {
                    tmpCnnPop.c_kernel[i][j][k] += tmpCnnPop.stepInGuass[0]
                                                   * randGauss(0,gaussRange);
                    makeInRange(tmpCnnPop.c_kernel[i][j][k],
                                           Min_kernel,Max_kernel);
                }
    if(randFloat()<=F_CNN) {
        tmpCnnPop.stepInGuass[0] += tmpCnnPop.stepInGuass[0] * randGauss(0,gaussRange);
        makeInRange(tmpCnnPop.stepInGuass[0],Min_kernel,Max_kernel);
    }
    //对偏置参数执行高斯变异
    for(i=0; i<tmpCnnPop.bias.size(); ++i)
        if(randFloat()<=F_CNN) {
            tmpCnnPop.bias[i] += tmpCnnPop.stepInGuass[1] * randGauss(0, gaussRange);
            makeInRange(tmpCnnPop.bias[i],Min_bias,Max_bias);
        }
    if(randFloat()<=F_CNN) {
        tmpCnnPop.stepInGuass[1] += tmpCnnPop.stepInGuass[1] * randGauss(0,gaussRange);
        makeInRange(tmpCnnPop.stepInGuass[1],Min_bias,Max_bias);
    }
    //对矩阵左边的神经元参数执行高斯变异
    for (i = 0; i < (mlDimension+1)*numPerCol; i++)
        if (randFloat() <= F_Matrix) {
            j = i/(mlDimension+1), k = i%(mlDimension+1);
            tmpMatPop.ww[j][k] += randGauss(0, 1);
            makeInRange(tmpMatPop.ww[j][k],-1,1);
        }
    //对矩阵部分执行多点变异
    for (i=0; i<chromLength; i++)
        if(randFloat()<=F_Matrix) {
            if(((i+1)%3)==0)	//每到第三个数变为操作符的索引
                tmpMatPop.chrom[i] ^= rand() % OP_NUMS;
            else
                tmpMatPop.chrom[i] ^= rand() % numPerCol;
        }
}
//从卷积网络得到输出作为矩阵网络的输入,计算个体_cnnPop与_pop结合后的适应值
float MyCNN::getFitValue(CNNIndividual& _cnnPop, Individual& _pop)
{
    //int t_start = clock();
    uint error = 0,errorBiggerThan10=0;		//记录标签识别错误个数
    uint i_input;
    for(i_input=0; i_input<cnnTrain.size(); ++i_input) {
        updatePerLayer(cnnTrain[i_input].data,_cnnPop);
        getCNNOut(matTrain[i_input].data);
        int tag = cnnTrain[i_input].tag;	//得到该个体的标签
        int getTag = getPEOut(_pop, matTrain[i_input].data);
        if(tag!=getTag) {
            if(getTag>9)
                ++errorBiggerThan10;
            else
                ++error;
        }
    }
    _cnnPop.fitValue = errorBiggerThan10*2.0f+error;
    if(_cnnPop.fitValue>cnnTrain.size())
        _cnnPop.fitValue = (float)cnnTrain.size();
    //cout<<clock()-t_start<<endl;
    //cin.get();
    return _cnnPop.fitValue;
}
