#include"CNN_Base.hpp"

const uint  CNN_Base::CS_NUM[6] = {5,5,10,10,20,11};
const float CNN_Base::Min_kernel = -1, CNN_Base::Max_kernel = 1;
const float CNN_Base::Min_bias = -1.5,  CNN_Base::Max_bias = 1.5;

CNN_Base::CNN_Base(uint _inputSize,uint _popSize)
{
	inputSize = _inputSize;
    cnnPop.resize(_popSize);
    constructCS();
}
CNN_Base::~CNN_Base() {}
//构造CS各层的参数向量,并初始化卷积核和偏置参数
void CNN_Base::constructCS()
{
    uint i,j,k,index;
    setSrand();	//设置随机种子
    for(index=0; index<cnnPop.size(); ++index) {
        //初始化卷积核. 范围是经验值,可以在后期调整
        for(i=0; i<kernelNum; ++i) {
            cnnPop[index].c_kernel[i].resize(kernelSize);
            for(j=0; j<kernelSize; ++j) {
                cnnPop[index].c_kernel[i][j].resize(kernelSize);
                for(k=0; k<kernelSize; ++k)
                    cnnPop[index].c_kernel[i][j][k] = randFloat(Min_kernel,Max_kernel);
            }
        }
        //初始化偏置参数.每个Map对应一个偏置. 范围是经验值,可以在后期调整
        cnnPop[index].bias.resize(61);
        for(i=0; i<cnnPop[index].bias.size(); ++i)
            cnnPop[index].bias[i] = randFloat(Min_bias,Max_bias);
        //初始化步长
        cnnPop[index].stepInGuass[0] = randFloat(Min_kernel,Max_kernel);
        cnnPop[index].stepInGuass[1] = randFloat(Min_bias,Max_bias);
    }
    try {
        //初始化C1
        int C1Size = inputSize-kernelSize+1;
        C1.resize(CS_NUM[0],vectorF2D(C1Size,vectorF(C1Size)));
        //初始化S2
        int S2Size = (C1Size+subSampleSize-1)/subSampleSize;
        S2.resize(CS_NUM[1],vectorF2D(S2Size,vectorF(S2Size)));
        //初始化C3
        int C3Size = S2Size-kernelSize+1;
        C3.resize(CS_NUM[2],vectorF2D(C3Size,vectorF(C3Size)));
        //初始化S4
        int S4Size = (C3Size+subSampleSize-1)/subSampleSize;
        S4.resize(CS_NUM[3],vectorF2D(S4Size,vectorF(S4Size)));
        //初始化C5
        int C5Size = 1;//将C5大小直接设置为1\\S4Size-kernelSize+1;
        C5.resize(CS_NUM[4],vectorF2D(C5Size,vectorF(C5Size)));
        //初始化S6
        int S6Size = (C5Size+subSampleSize-1)/subSampleSize;
        S6.resize(CS_NUM[5],vectorF2D(S6Size,vectorF(S6Size)));
    } catch(...) {
        throw "输入图大小inputSize设置不正确，无法构建完整卷积网络，请检查!";
    }
}
//增加新的输入图
void CNN_Base::addTrain(const vectorF2D& grayData, int tag)
{
    if(grayData.size()<inputSize || grayData[0].size()<inputSize) {
        cout<< "wrong size of input grayData,input grayData must be "<<inputSize<<"*"<<inputSize<<endl;
        return;
    }
    S_CNNInput input = {grayData,tag};
    cnnTrain.push_back(input);
}
//载入输入数据，被loadTrain和loadTest调用
int CNN_Base::loadInput(const char* file,int size,vector<S_CNNInput> &dataVec,bool haveTag)
{
    ifstream loadFile(file);
    if(loadFile.is_open()) {
        if(size>0)
			dataVec.reserve(size);
        uint i,j,k;
        bool success = true;	//如果某个数据读取错误，则失败
        S_CNNInput input = {vectorF2D(inputSize,vectorF(inputSize)),0};
        for(i = 0; size < 1 || i < size; ++i) {
            for(j = 0; j < inputSize && success; ++j)
                for(k = 0; k < inputSize && success; ++k)
                    if(!(loadFile>>input.data[j][k])) {
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
    }else
		return 0;
}
//从文件载入训练数据
void CNN_Base::loadTrain(const char* file,int size)
{
    if(file==0 || file[0]==0 || size==0){
		cout<<"\r请正确设置训练集文件路径和个数"<<endl;
        throw "请正确设置训练集文件路径和个数";
    }
	int i = loadInput(file,size,cnnTrain,1);
	if(i>0)
        cout<<"\r\t\t\t\t\t\r>>成功载入"<<i<<"组训练集"<<endl;
	else
		cout<<"\r\t\t\t\t\t\r文件‘"<<file<<"’打开失败！"<<endl;
}
//从文件载入测试数据
void CNN_Base::loadTest(const char* file,int size,bool haveTag)
{
    if(file==0 || file[0]==0 || size==0){
		cout<<"\r请正确设置测试集文件路径和个数"<<endl;
        throw "请正确设置测试集文件路径和个数";
    }
	int i = loadInput(file,size,cnnTest,haveTag);
	if(i>0)
        cout<<"\r\t\t\t\t\t\r>>成功载入"<<i<<"组测试集"<<endl;
	else
		cout<<"\r\t\t\t\t\t\r文件‘"<<file<<"’打开失败！"<<endl;
}
//用inputData作为输入更新C1~S6每层feature Map的值
void CNN_Base::updatePerLayer(const vectorF2D& inputData, const CNNIndividual& indiv)
{
    uint i;
    //更新C1层,C1层有CS_NUM[0](=5)个特征图Map,取前5个(即0~4号)卷积核对输入图inputData进行卷积
    for(i=0; i<C1.size(); ++i)//用i号卷积核与输入图卷积再加上偏置得到C1[i]
        convoluteMap(C1[i],inputData,indiv.c_kernel[i],indiv.bias[i]);
    //更新S2层,S2层有CS_NUM[1](=5)个Map,将C1中相对应特征图的subSampleSize*subSampleSize邻域抽样,再加偏置
    for(i=0; i<S2.size(); ++i)
        subSampleMap(S2[i],C1[i],indiv.bias[CS_NUM[0]+i],SSM_Max);
    //更新C3层,C3层有CS_NUM[2](=10)个特征图Map,取5~14号卷积核与S2层相应Map进行卷积
    updateC3Layer(indiv.c_kernel,indiv.bias);
    //更新S4层,S4层有CS_NUM[3](=10)个Map,将C3中相对应特征图的subSampleSize*subSampleSize邻域抽样,再加偏置
    static const uint CS_NUM_SUM = CS_NUM[0]+CS_NUM[1]+CS_NUM[2];
    for(i=0; i<S4.size(); ++i)
        subSampleMap(S4[i],C3[i],indiv.bias[CS_NUM_SUM+i],SSM_Max);
    //更新C5层,C5层有CS_NUM[4](=20)个特征图Map,取0~20号卷积核分别与S4层每个Map进行卷积,C5层每个Map只有一个值
    updateC5Layer(indiv.c_kernel,indiv.bias);
    //更新S6层,S6层有CS_NUM[5](=11)个特征图Map,S6中每个输出的特征值从C5中随机取两个Map取偏置求和得到
    //updateS6Layer(indiv.bias);
}
//对可训练参数(卷积核和偏置)执行变异
void CNN_Base::kernelBiasMutate()
{

}
//更新C3层,S2到C3的映射有点复杂,在不同的情况下可能会被重写。C3层有CS_NUM[2](=10)个特征图Map,
//取5~14号卷积核与S2层相应Map进行卷积。下面的计算只适用S2有5个Map,C3有10个Map的情况
void CNN_Base::updateC3Layer(const kernelType* c_kernel,const biasType& bias)
{
    uint i;
    static const uint biasStart = CS_NUM[0]+CS_NUM[1];
    vectorF2D tmp(C3[0].size(),vectorF(C3[0][0].size(),0)),tmp2[4];
    tmp2[0] = tmp2[1] = tmp2[2] = tmp2[3] = tmp;//tmp2[0~3]已初始化大小
    for(i=0; i<4; ++i) {
        convoluteMap(tmp2[0],S2[i]  ,c_kernel[5+i], bias[biasStart+i]);
        convoluteMap(C3[i]  ,S2[i+1],c_kernel[5+i], bias[biasStart+i]);
        addVector(C3[i],tmp2[0]);
		sigmoidMap(C3[i]);
    }
    for(i=0; i<3; ++i) {
        convoluteMap(tmp2[0],S2[i]  ,c_kernel[9+i],bias[biasStart+4+i]);
        convoluteMap(tmp2[1],S2[i+1],c_kernel[9+i],bias[biasStart+4+i]);
        convoluteMap(C3[4+i],S2[i+2],c_kernel[9+i],bias[biasStart+4+i]);
        addVector(C3[4+i],tmp2[0],tmp2[1]);
		sigmoidMap(C3[4+i]);
    }
    for(i=0; i<2; ++i) {
        convoluteMap(tmp2[0],S2[i]  ,c_kernel[12+i],bias[biasStart+7+i]);
        convoluteMap(tmp2[1],S2[i+1],c_kernel[12+i],bias[biasStart+7+i]);
        convoluteMap(tmp2[2],S2[i+2],c_kernel[12+i],bias[biasStart+7+i]);
        convoluteMap(C3[7+i],S2[i+3],c_kernel[12+i],bias[biasStart+7+i]);
        addVector(C3[7+i],tmp2[0],tmp2[1],tmp2[2]);
		sigmoidMap(C3[7+i]);
    }
    convoluteMap(tmp2[0],S2[0],c_kernel[14],bias[biasStart+9]);
    convoluteMap(tmp2[1],S2[1],c_kernel[14],bias[biasStart+9]);
    convoluteMap(tmp2[2],S2[2],c_kernel[14],bias[biasStart+9]);
    convoluteMap(tmp2[3],S2[3],c_kernel[14],bias[biasStart+9]);
    convoluteMap(C3[9]  ,S2[4],c_kernel[14],bias[biasStart+9]);
    addVector(C3[9],tmp2[0],tmp2[1],tmp2[2],tmp2[3]);
	sigmoidMap(C3[9]);
}
//通过全连接形式更新C3层,每一个C3层图对每一个S2层的图都需要一个卷积核
void CNN_Base::updateC3AllConnect(const kernelType* c_kernel,const biasType& bias)
{
	uint i,j, s2_Size = S2.size(), c3_size = C3.size();
    static const uint biasStart = CS_NUM[0]+CS_NUM[1];
	static vectorF2D tmp(C3[0].size(),vectorF(C3[0][0].size()));
	for(i=0;i<c3_size;++i) {
		zeroVector(C3[i]);
		for(j=0;j<s2_Size;++j) {
			convoluteMap(tmp,S2[j],c_kernel[i+CS_NUM[2]*j+CS_NUM[0]], bias[biasStart+i]);
			addVector(C3[i],tmp);
		}
		sigmoidMap(C3[i]);
	}
}
//更新C5层,C5层有CS_NUM[4](=20)个特征图Map,取15~34号卷积核分别与S4层每个Map进行卷积,C5层每个Map只有一个值
void CNN_Base::updateC5Layer(const kernelType* c_kernel,const biasType& bias)
{
    uint i,j;
    static const uint CS_NUM_SUM = CS_NUM[0]+CS_NUM[1]+CS_NUM[2]+CS_NUM[3];
    static const uint kernelStart = CS_NUM[0] + CS_NUM[2];
    float c5Elem;	//这里需要保证C5中Map都是1*1大小,即一个float数
    for(i=0; i<C5.size(); ++i) {
        c5Elem = 0;
        for(j=0; j<S4.size(); ++j) { //用对应卷积核与S4中每个图卷积再加上偏置得到C5[i]
            convoluteMap(C5[i],S4[j],c_kernel[i+kernelStart],bias[CS_NUM_SUM+i]);
            c5Elem += C5[i][0][0];
        }
        C5[i][0][0] = c5Elem;
    }
}
//更新S6层,S6层有CS_NUM[5](=11)个特征图Map,S6中每个输出的特征值从C5中随机取两个Map取偏置求和得到
void CNN_Base::updateS6Layer(const biasType& bias)
{
    static const uint CS_NUM_SUM = CS_NUM[0]+CS_NUM[1]+CS_NUM[2]+CS_NUM[3]+CS_NUM[4];
    for(uint i=0; i<S6.size(); ++i)	//这里需要保证S6中Map都是1*1大小
        S6[i][0][0] = 2 * bias[CS_NUM_SUM+i] + C5[i][0][0] + C5[(i+S6.size())%C5.size()][0][0];
}
//把一个特征图input与卷积核kernel执行卷积操作再加偏置bias后,得到对应的特征映射图output
void CNN_Base::convoluteMap(vectorF2D& output,const vectorF2D& input,const kernelType& kernel,float _bias)
{
    uint j,k;
    for(j=0; j<output.size(); ++j)
        for(k=0; k<output[j].size(); ++k)
            output[j][k] = getConvoluteOut(input,kernel,j,k) + _bias;//从起点为(j,k)的点开始做卷积
}
//对一个特征图input通过mode方式下采样后再加偏置bias,得到对应的特征提取图output
void CNN_Base::subSampleMap(vectorF2D& output,const vectorF2D& input,float _bias,int mode)
{
    uint j,k;
    for(j=0; j<output.size(); ++j)
        for(k=0; k<output[j].size(); ++k)
            output[j][k] = sigmoid(getSubSampleOut(input,j*subSampleSize,k*subSampleSize,mode) + _bias);
}
//把out与vec1相加保存到out
void CNN_Base::addVector(vectorF2D& output,const vectorF2D& vec1)
{
    uint i,j;
    for(i=0; i<output.size() && i<vec1.size(); ++i)
        for(j=0; j<output[i].size() && j<vec1[i].size(); ++j)
            output[i][j] += vec1[i][j];
}
//将各层的特征图打印到文件中
void CNN_Base::printCSToFile(const char* file)
{
    ofstream outFile(file);
    if(outFile.is_open()) {
        uint i,j,k;
        for(i=0; i<C1.size(); ++i) {
            outFile<<"C1-"<<i<<endl;
            for(j=0; j<C1[0].size(); ++j) {
                for(k=0; k<C1[0][0].size(); ++k)
                    outFile<<setw(4)<<C1[i][j][k]<<" ";//<<setw(5)
                outFile<<endl;
            }
            outFile<<"————————————————————————————"<<endl;
        }
        for(i=0; i<S2.size(); ++i) {
            outFile<<"S2-"<<i<<endl;
            for(j=0; j<S2[0].size(); ++j) {
                for(k=0; k<S2[0][0].size(); ++k)
                    outFile<<setw(4)<<S2[i][j][k]<<" ";//<<setw(5)
                outFile<<endl;
            }
            outFile<<"————————————————————————————"<<endl;
        }
        for(i=0; i<C3.size(); ++i) {
            outFile<<"C3-"<<i<<endl;
            for(j=0; j<C3[0].size(); ++j) {
                for(k=0; k<C3[0][0].size(); ++k)
                    outFile<<setw(4)<<C3[i][j][k]<<" ";//<<setw(5)
                outFile<<endl;
            }
            outFile<<"————————————————————————————"<<endl;
        }
        for(i=0; i<S4.size(); ++i) {
            outFile<<"S4-"<<i<<endl;
            for(j=0; j<S4[0].size(); ++j) {
                for(k=0; k<S4[0][0].size(); ++k)
                    outFile<<setw(4)<<S4[i][j][k]<<" ";//<<setw(5)
                outFile<<endl;
            }
            outFile<<"————————————————————————————"<<endl;
        }
        for(i=0; i<C5.size(); ++i) {
            outFile<<"C5-"<<i<<endl;
            for(j=0; j<C5[0].size(); ++j) {
                for(k=0; k<C5[0][0].size(); ++k)
                    outFile<<setw(4)<<C5[i][j][k]<<" ";//<<setw(5)
                outFile<<endl;
            }
            outFile<<"————————————————————————————"<<endl;
        }
        for(i=0; i<S6.size(); ++i) {
            outFile<<"S6-"<<i<<endl;
            for(j=0; j<S6[0].size(); ++j) {
                for(k=0; k<S6[0][0].size(); ++k)
                    outFile<<setw(4)<<S6[i][j][k]<<" ";//<<setw(5)
                outFile<<endl;
            }
            outFile<<"————————————————————————————"<<endl;
        }
        outFile<<"————————————————————————————————————————————————————————"<<endl;
        for(uint index=0; index<cnnPop.size(); ++index) {
            for(i=0; i<kernelNum; ++i) {
                outFile<<"cnnPop-"<<index<<"  kernel-"<<i<<endl;
                for(j=0; j<cnnPop[index].c_kernel[i].size(); ++j) {
                    for(k=0; k<cnnPop[index].c_kernel[i][j].size(); ++k)
                        outFile<<cnnPop[index].c_kernel[i][j][k]<<"  ";
                    outFile<<endl;
                }
                outFile<<"————————————————————————————"<<endl;
            }
            outFile<<"cnnPop["<<index<<"].stepInGuass[0] = "<<cnnPop[index].stepInGuass[0]<<endl;
        }
        outFile<<"————————————————————————————————————————————————————————"<<endl;
        for(uint index=0; index<cnnPop.size(); ++index) {
            outFile<<"bias-"<<index<<endl;
            for(i=0; i<cnnPop[index].bias.size(); ++i)
                outFile<<cnnPop[index].bias[i]<<"  ";
            outFile<<endl;
            outFile<<"cnnPop["<<index<<"].stepInGuass[1] = "<<cnnPop[index].stepInGuass[1]<<endl;
        }
        outFile<<"————————————————————————————————————————————————————————"<<endl;
        for(i=0; i<cnnTrain.size(); ++i) {
            outFile<<"cnnTrain-"<<i<<endl;
            for(j=0; j<cnnTrain[i].data.size(); ++j) {
                for(k=0; k<cnnTrain[i].data[j].size(); ++k)
                    outFile<<setw(3)<<cnnTrain[i].data[j][k]<<" ";
                outFile<<endl;
            }
            outFile<<"   "<<cnnTrain[i].tag<<endl;
            outFile<<"————————————————————————————"<<endl;
        }
        outFile.close();
        cout<<"已输出到文件"<<file<<"！"<<endl;
    }
}
