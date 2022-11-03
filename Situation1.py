# %%
# 为Gorubi+启发式算法
# @author Niki
# version 2022.7.19_v8 添加了承载量约束
# version 2022.7.19_v9 添加了承载量约束

# %% [markdown]
# ### 配置环境

# %%
import imp
from gurobipy import *
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import os
import time
import collections
import copy

# %% [markdown]
# ### 读取数据

# %%
node = 32
T= 600
v_flow = 750
Max_link_cap = v_flow * T
# -*- coding: utf-8 -*-
# fnum = 50

# %%
base_path = os.getcwd()
data_path = base_path + '/'
data_flow_path = './flow140/'

# %%
flow_size = pd.read_excel(os.path.join(data_path,'32节点140条周期流信息.xlsx'), usecols=[1,7], sheet_name = 'Sheet1')
print(flow_size)
flow_T_size = [T/flow_size.iloc[i][0] * flow_size.iloc[i][1]  for i in range(len(flow_size))] #公周期流的大小
# print(flow_T_size)

# %%
link_oneside = pd.read_csv(data_path + 'unidirLink32.csv', encoding = 'unicode_escape', header = 0, index_col = 0)
# print(link_oneside)
num_link = link_oneside.shape[0]
# print(num_link)
temp = ['(%s'%(a) + ', %d'%(b) + ')' for [a,b] in link_oneside.values]
# print(temp)
link_dict = dict((b,a) for a,b in enumerate(temp))
link_dict_conv = dict(enumerate(temp))
# print(link_dict)

# %%
path = pd.read_csv(data_path + 'path32.csv', index_col = 0)
# print(path)
print(path.shape)

# %%
link_forward = pd.read_csv(data_path + 'forwardLink32.csv', index_col = 0)
# print(link_forward)
print(link_forward.shape[0])

# %%
link_forward_useful = [[] for k in range(link_forward.shape[0])]
for i in range(link_forward.shape[0]):
    for ch in link_forward.iloc[i]:
        if not pd.isnull(ch):
            link_forward_useful[i].append(link_dict[ch])
        else:
            break
# print(link_forward_useful)
print(len(link_forward_useful))

# %%
os.chdir(data_flow_path) # 设置工作目录
file_chdir = os.getcwd() # 获得工作目录

# %%
filename_list = os.listdir(file_chdir) 
"""os.listdir(path) 扫描路径的文件，将文件名存入存入列表"""
for file in filename_list:
    used_fileName,extension = os.path.splitext(file)
    new_file = used_fileName.split('_')[0] + extension
    os.rename(file, new_file)
    # print("文件%s重命名成功,新的文件名为%s" %(used_fileName,new_file))
print(len(filename_list))

# %%

filename_npy = [] # 文件名列表
file_npy = [] # 数据列表
file_path = [] # 路由列表
for root,dirs,files in os.walk(file_chdir): # os.walk会便利该目录下的所有文件
    files.sort(key =lambda x:int(x[4:-4]))
    for file in files:
        #print(file)
        if os.path.splitext(file)[-1] == '.csv': # 判断文件格式是否符合csv格式
            filename_npy.append(file.split('.')[0][4:])
            #print(filename_npy)
            file_npy.append(np.array(pd.read_csv(file, index_col = 0, sep = None, engine='python'))) # 存储数据，可以改成“字典”形式
            file_path.append(pd.read_csv(file, usecols=[0]).values)
file_npy1 = {}
for v,i in enumerate(file_npy):
    file_npy1[v] = i

# print(file_npy1) 
# flow_data = file_npy1 #存的字典
flow_data1 = file_npy1 # data就是所有数据的存储
# print(flow_data1)

# %% [markdown]
# 

# %%
os.chdir(base_path) # 设置工作目录
# print(flow_data)

# %%
flow_num = len(flow_data1)  #流数
print(flow_num)

# %% [markdown]
# ### 问题设置

# %%
class Problem:
    def __init__(self, node, flow_num, link_forward_useful, flow_data, num_link, Max_link_cap, file_flow_size ) -> None:
        self.node = node                                #节点数(矩阵大小)
        self.flow_num = flow_num                        #流数量
        self.flow_data = flow_data                      #流数据（路由对应单向链路信息）
        self.file_flow_size = file_flow_size            #流大小
        self.link_forward_useful = link_forward_useful  #可行路径矩阵
        self.num_link = num_link                        #容量矩阵  单向链路集合
        self.M = flow_num   #修改
        self.Max_link_cap =Max_link_cap
    
    def create_model(self):
        self.x = [[] for _ in range(self.flow_num)]
        self.y = [[] for _ in range(self.num_link)]
        self.model = gp.Model("MinMaxProb")
        self.__set_vars()
        self.__set_contrs()
        self.__set_objective()
    
    def solve(self, flag = 0):
        #self.model.Params.PoolGap = 0.25                    #找到的可行解与最优解的Gap值
        self.model.Params.MIPGap = 0.01
        # self.model.Params.TimeLimit = 3600                  #求解时长限制
        # self.model.Params.PoolSearchMode = 2                #搜索模式
        self.model.Params.OutputFlag = flag
        self.model.optimize()
        
    def __set_vars(self) -> None:
        for k in range(self.flow_num):
            for i in range(len(self.flow_data[k])):
                self.x[k].append(self.model.addVar(obj = 1 , lb = 0, ub = 1, vtype = GRB.INTEGER, name = 'zf' + str(k) + 'p'+ str(i)))
        for k in range(self.num_link):
            for i in range(len(self.link_forward_useful[k])):
                self.y[k].append(self.model.addVar(obj = 1 , lb = 0, ub = 1, vtype = GRB.INTEGER, name = 'b_link' + str(i) + 'to'+ str(k)))
        self.target = self.model.addVar(obj = 1, lb = 0, ub = GRB.INFINITY, vtype = GRB.INTEGER, name = 'z')
        
    def __set_contrs(self) -> None:
        self.model.addConstrs(gp.quicksum(self.x[k][i] for i in range(len(self.flow_data[k])) ) == 1 for k in range(self.flow_num))
        self.model.addConstrs(gp.quicksum(self.x[k][i] * self.flow_data[k][i][self.link_forward_useful[b][a]] * self.flow_data[k][i][b] for k in range(self.flow_num) for i in range(len(self.flow_data[k])))\
            <= self.y[b][a] * self.M for b in range(self.num_link) for a in range(len(self.link_forward_useful[b])))
        self.model.addConstrs(gp.quicksum(self.y[k][i] for i in range(len(self.link_forward_useful[k]))) <= self.target for k in range(self.num_link))
        self.model.addConstrs(gp.quicksum(self.x[k][i] * self.file_flow_size[k] * self.flow_data[k][i][j]  for k in range(self.flow_num) for i in range(len(self.flow_data[k]))) \
            <= self.Max_link_cap  for j in range(self.num_link) if self.link_forward_useful[j] != [])

    def __set_objective(self) -> None:
        self.model.setObjective(self.target, sense = GRB.MINIMIZE)
    
    def get_best_solution(self):
        temp = 0
        solution = []
        for k in range(self.flow_num):
            for i in range(len(self.flow_data[k])):
                if self.model.getVars()[temp].x == 1.0:
                    solution.append([k,i])
                temp += 1
        return solution
    
    def get_obj(self):
        return self.model.ObjVal
    
    def print_status(self):
        status = self.model.status
        if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            print('The model cannot be solved because it is infeasible or unbounded')
            return
        if status != GRB.OPTIMAL:
            print('Optimization was stopped with status ' + str(status))
            return
        print("master objective value: {}".format(self.model.ObjVal))   # 输出目标函数的值


# %%

class Heuristic:
    def __init__(self, node, flow_num, link_forward, flow_data, num_link, file_flow_size) -> None: # 初始化
        self.node = node                                #节点数(矩阵大小)
        self.flow_num = flow_num                        #流数量
        self.flow_data = flow_data                      #流数据（路由对应单向链路信息）
        self.link_forward = link_forward                #可行路径矩阵
        self.num_link = num_link                        #容量矩阵 单向链路总数
        # self.delay = delay                              #时延矩阵
        self.M = flow_num
        self.file_flow_size = file_flow_size            #当前资源占有量

    
    def create_model(self, solution_base):
        self.x = collections.defaultdict(lambda : [])
        self.capacity_cur = collections.defaultdict(lambda : 0)
        # self.x = [[] for _ in range(self.num_link)] #链路总数*1
        # self.y = [0] * self.num_link  # 1 *链路总数
        self.y = collections.defaultdict(lambda : 0)
        self.solution_base = solution_base
        for f in range(self.flow_num):
            for k in range(self.num_link): 
                if flow_data1[f][solution_base[f][1]][k] == 1:  #第f条流的可行解是否经过第k条链路
                    #self.x[k].append(f)
                    #print(self.x[k],k)
                    self.x[k] = self.x[k] +[f]
                    self.capacity_cur[k] = self.capacity_cur[k] + self.file_flow_size[f]    #求解当前网络资源量
                    # print(self.x,k)
        for k in range(self.num_link):
            for pre in self.link_forward[k]:
                for f in self.x[k]:
                    if f in self.x[pre]:
                        self.y[k] += 1  #第k条链路的汇聚数
                        break
        self.target = max(self.y.values())  #网络的最大汇聚数
        
    def remove_model(self, flow_num, x,  y, capacity_cur):
        self.x = copy.deepcopy(x)
        #self.y = [0] * self.num_link
        self.y = copy.deepcopy(y)
        self.capacity_cur = copy.deepcopy(capacity_cur)

        for k in range(self.num_link):
            for pre in self.link_forward[k]:
                if (set(self.x[k]) & set(self.x[pre])) == {flow_num}:
                    self.y[k] -= 1  #第k条链路的汇聚数
                    break
        
        #print('1', self.x)
        for k in self.x.keys():
            if flow_num in self.x[k]:
                self.x[k].remove(flow_num)
                self.capacity_cur[k] = self.capacity_cur[k] - self.file_flow_size[flow_num]

    def input_modle(self,solution_base, flow_num, x,  y, capacity_cur):

        self.x = copy.deepcopy(x)
        #self.y = [0] * self.num_link
        self.y = copy.deepcopy(y)
        self.capacity_cur = copy.deepcopy(capacity_cur)
        self.solution_base = copy.deepcopy(solution_base)
        
        for k in range(self.num_link): 
            if flow_data1[flow_num][self.solution_base[flow_num][1]][k] == 1:  #第f条流的可行解是否经过第k条链路
                # self.x[k].append(f)
                self.x[k] = self.x[k] +[flow_num]
                self.capacity_cur[k] = self.capacity_cur[k] + self.file_flow_size[flow_num]  # #更新当前网络资源量

        for k in range(self.num_link):
            for pre in self.link_forward[k]:
                if (set(self.x[k]) & set(self.x[pre])) == {flow_num}:
                    # print(flow_num)
                    self.y[k] += 1  #第k条链路的汇聚数
                    break

        self.target = max(self.y.values())  #网络的最大汇聚数

    def get_cons(self):
        #print('3', self.x)
        #print('target', self.target)
        return self.target
    
    def get_obj(self):
        #print('4', self.x)
        self.ans = 0
        for k in range(self.num_link):
            if self.y[k] > 1:
                self.ans += self.y[k]
        #print('ans', self.ans)
        return self.ans
    
    def get_solution(self):
        return self.solution_base
        
    def get_x(self):
        return self.x

    def get_y(self):
        return self.y
    
    def get_cap(self):
        return self.capacity_cur
    def get_min(self):  #汇聚数为1的和二的加权总和
        onesum = 0
        for  i in self.y.values():
            if 1 <= i < self.target:
                onesum += 1
        return onesum

# %% [markdown]
# ### 测试

# %%
model = gp.Model("HEUR")

# %% [markdown]
# ### 迭代

# %%
MAX_ITER_TIMES = 10000
Base = Problem(node, flow_num, link_forward_useful, flow_data1, num_link, Max_link_cap, flow_T_size)
Base.create_model()

# %%
# CG_RR.to_int()
Base.solve(flag=1)

# %%
Base.print_status()

# %%
ans_z = Base.model.getVars()
#print(ans_z)
ans_t = Base.get_obj()
print(ans_t)

# %%
ori_solution = Base.get_best_solution()
solution = np.array(Base.get_best_solution())
# print(ori_solution)

# %%
import copy
import pickle

def deepsave(data):
    return pickle.loads(pickle.dumps(dict(data)))

HEUR = Heuristic(node, flow_num, link_forward_useful, flow_data1, num_link, flow_T_size)
base_solution = solution.copy()
HEUR.create_model(solution)
cons = HEUR.get_cons()
ans = HEUR.get_obj()
cap = sum(HEUR.get_cap().values())
capMax = max(HEUR.get_cap().values())
jiaquan  = HEUR.get_min()
# x2 = deepsave(HEUR.get_x())
# y2 = deepsave(HEUR.get_y())
x2 = copy.deepcopy(HEUR.get_x())
y2 = copy.deepcopy(HEUR.get_y())
capacity_cur2 = copy.deepcopy(HEUR.get_cap())  #网络资源利用情况
if max(capacity_cur2.values())> Max_link_cap:
    print('基本可行解不可行，超出最大承载力')
# print(x,y)
k = 0

# %%
print(ans)  #可行解的总汇聚数
print(cons) #可行解的最大汇聚数
print(cap) #可行解的网络资源总利用情况
print(capMax) #可行解的网络资源最大利用情况
print(jiaquan)
for i in capacity_cur2.keys():
    if capacity_cur2[i] == 193125.0:
        print(x2[v])

# %%
#一直迭代
flag = 1
start_pos = -1
ori_ans = ans
ori_cap = cap
ori_max = capMax
ori_jiqauqn = jiaquan
solution_temp = base_solution.copy()
flow_data2 = dict(sorted(flow_data1.items(),key = lambda x:len(x[1]),reverse=True))
while flag:
    flag = 0
    #for i in range(flow_num):
    for i in flow_data2.keys():
        f = i
        # f = (i + start_pos + 1) % flow_num
        HEUR.remove_model(f, x2, y2, capacity_cur2)

        x1 = copy.deepcopy(HEUR.get_x())
        y1 = copy.deepcopy(HEUR.get_y())
        capacity_cur1 = copy.deepcopy(HEUR.get_cap())
        # print(i, max(capacity_cur1.values()))

        for p in range(len(flow_data1[f])):
            solution = base_solution.copy()
            solution[f][1] = p
            HEUR.input_modle(solution, f, x1, y1, capacity_cur1)
            total_cap = sum(HEUR.get_cap().values())
            max_cap = max(HEUR.get_cap().values())
            cur_jiaquan = HEUR.get_min()
            #print(i, p, HEUR.get_obj(), ans,cur_jiaquan,jiaquan)

            if max_cap > Max_link_cap:
                print(str(max_cap) + '超出最大承载力')
                continue
            if cons < HEUR.get_cons():
                continue
            #print(max_cap, cap)
            if cur_jiaquan > jiaquan or (cur_jiaquan == jiaquan  and total_cap < cap):
                print('yes')
            #if (HEUR.get_obj() < ans  and total_cap <= cap) or ((HEUR.get_obj() <= ans  and total_cap < cap)):
            #if (HEUR.get_obj() < ans  and max_cap <= capMax) or ((HEUR.get_obj() <= ans  and max_cap < capMax)):
                ans = HEUR.get_obj()
                cap = total_cap
                capMax = max_cap
                jiaquan  = cur_jiaquan
                print(jiaquan)
                solution_temp = copy.deepcopy(np.array(HEUR.get_solution()))

    
                x2 = copy.deepcopy(HEUR.get_x())
                y2 = copy.deepcopy(HEUR.get_y())
                capacity_cur2 = copy.deepcopy(HEUR.get_cap())
                temp = p
                
        if jiaquan > ori_jiqauqn or (jiaquan == ori_jiqauqn and cap < ori_cap):
        #if (ans < ori_ans and capMax <= ori_max) or (ans <= ori_ans and capMax < ori_max):
            k += 1
            print([k, ans, capMax/750, f, jiaquan, temp])
            ori_ans = ans
            ori_cap = cap
            ori_max = capMax
            ori_jiqauqn = jiaquan
            base_solution = copy.deepcopy(solution_temp)
            #print(base_solution)
            flag = 1
            start_pos = f
            break

# %%
print(k)
print(base_solution)
result = []
for i in range(flow_num):
    result.append([filename_npy[i],file_path[i][base_solution[i][1]][0]])
# print(result)
pd.DataFrame(result).to_csv('方案一solution_heur_32_140_unique600_situ1_2.csv', index=False, header=False)


