/*
 * Copyright (c) 2015
 * 闻波, webary, HBUT 
 * reference ：https://github.com/webary/MyCNN/
 * 	           http://www.cnblogs.com/webary/
 */

#pragma once
#ifndef  _INI_Util_HPP_
#define  _INI_Util_HPP_
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
using namespace std;

//读取配置文件类
class INI_Util {
    typedef struct {
        string key;		//关键字
        string value;	//键值
    } Record;
    typedef struct {
        string node;		//节点名
        vector<Record> r_set;	//记录集合
    } Group;
    vector<Group> conf;

    string defaultNode;	//默认查找节点
    string state;		//搜索状态,查找失败是存储失败原因
	string lastFileName;//上一次载入的文件名

public:
#define For_each(it,vec) for(auto it=vec.begin();it<vec.end();++it)

    INI_Util(const string iniFileName="",const string _defaultNode="") {
        state = lastFileName = "";
        setDefaultNode(_defaultNode);
        if(iniFileName!="")
			loadINI(iniFileName);
    }

	void loadINI(const string &iniFileName) {
		if (iniFileName != "")
			lastFileName = iniFileName;
		if (lastFileName == "")
			return;
		ifstream readINI(lastFileName.c_str());
		if (readINI.is_open()) {
			//清除以前的信息
			For_each(it, conf)
				it->r_set.clear();
			conf.clear();
			string line;
			string::size_type pos;
			string tmpNode = "";
			while (getline(readINI, line)) {
				trim(line);
				if (line == "")
					continue;
				if (line[0] == '[' && line[line.length() - 1] == ']') {
					tmpNode = line.substr(1, line.length() - 2);
					continue;
				}
				pos = line.find('=');
				if (pos == line.npos)
					continue;
				string tmpKey = line.substr(0, pos);
				string tmpValue = line.substr(pos + 1);
				trim(tmpNode);
				trim(tmpKey);
				trim(tmpValue);
				Record record = { tmpKey,tmpValue };
				Group group = { tmpNode,vector<Record>() };
				bool found = false;
				For_each(it, conf) {
					if (it->node == tmpNode) {
						it->r_set.push_back(record);
						found = true;
						break;
					}
				}
				//当前数据没有存储该组
				if (!found) {
					conf.push_back(group);
					conf.rbegin()->r_set.push_back(record);
				}
			}
			readINI.close();
		}
	}

	string getRecord(const string &key, const string node = "") {
		if (key == "") {
			state = "key is invalid!";
			return "";
		}
		string findNode = node;
		if (node == "")
			findNode = defaultNode;
		state = "Node '" + findNode + "' not found!";
		For_each(it, conf) {
			if (it->node == findNode) {
				state = "key '" + key + "' not found!";
				For_each(it2, it->r_set) {
					if (it2->key == key) {
						state = "key '" + key + "' find out!";
						return it2->value;
					}
				}
				break;
			}
		}
		return "";
	}
	
	void showAllRecord() {
		For_each(it, conf) {
			cout << "[" << it->node << "]" << endl;
			For_each(it2, it->r_set)
				cout << it2->key << " = " << it2->value << endl;
			cout << endl;
		}
	}

    const string& getState()const {
        return state;
    }

    void setDefaultNode(const string &node) {
        defaultNode = node;
    }

	bool isNodeExist(const string &node) {
		if (node.empty())
			return false;
		For_each(it, conf)
			if (it->node == node)
				return true;
		return false;
	}

    const string& getDefaultNode()const {
        return defaultNode;
    }

	const string& getLastFileName()const {
		return lastFileName;
	}
private:
    //去除首尾的空格和\t
    static const string& trim(string &s) {
        if(s.empty())
            return s;
        s.erase(0,s.find_first_not_of(" \t"));
        s.erase(s.find_last_not_of(" \t") + 1);
        return s;
    }
};

#endif // _INI_Util_HPP_
