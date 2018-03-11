#!/usr/bin/env python
# -*- coding: utf-8 
# 参考网络代码

# 随机森林，划分为5份，决策树使用基尼指数法进行属性划分

import io
from csv import reader
import requests #安装命令 pip install requests
from random import randrange

################# 构造随机森林的一些方法  结束 #################

# CSV文件加载
def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset

# 除了标签列（最后一列），其他列都转换为float类型,在加载
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# 把数据集分为 k 份 
def cross_validation_split(dataset, n_folds):
    dataset_split = list() # 生成空列表
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy)) # 采用随机删除方法
            fold.append(dataset_copy.pop(index)) # #使用.pop()把里边的元素都删除（相当于转移），这k份元素各不相同。
        dataset_split.append(fold)
    return dataset_split

#  二分类每列，把数据集拆分为 属性集和标签集
def test_split(index, value, dataset):
    left, right = list(), list() #初始化两个空列表
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right #返回两个列表，每个列表以value为界限对指定行（index）进行二分类。

# 使用gini系数来获得最佳分割点，决策树
def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini

# 为数据集选择最佳分割点
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            # 2分类数据集
            groups = test_split(index, row[index], dataset)
            # 获取基尼基数
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    print ({'index':b_index, 'value':b_value})
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# 生成决策树
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# 构建决策树
def build_tree(train, max_depth, min_size):
    root = get_split(train) #获得最好的分割点,下标值，groups
    split(root, max_depth, min_size, 1)
    return root

# 使用决策树预测测试集的值
def predict(node, row):
    if row[node['index']] < node['value']: #用测试集来代入训练的最好分割点，分割点有偏差时，通过搜索左右叉树来进一步比较。
        if isinstance(node['left'], dict): #如果是字典类型，执行操作
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
        
# 决策树，也就是说这次集成学习是在决策树基础上进行的
def decision_tree(train, test, max_depth, min_size):
    # 使用训练集构建一个决策树
    tree = build_tree(train, max_depth, min_size)
    print(tree)
    predictions = list()
    # 使用决策树验证每个测试集的值
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)

# 计算正确率
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0 # #这个是二值分类正确性的表达式

# 使用交叉验证拆分来评估算法
# dataset, algorithm=decision_tree, n_folds=5, max_depth=5, min_size=10
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds) #把数据分为n_folds份
#     print(folds)
    scores = list() # 空list
    
    ################## 交叉验证法 ###################
    for fold in folds: # 循环每一个自助采样集
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
    ################## 交叉验证法 ###################
    
        # train_set 是移除 fold 后的集合
        # test_set 是 fold集合
        # algorithm=decision_tree，去调动决策树模型
        # 生成 5个 训练集的预测值
        predicted = algorithm(train_set, test_set, *args)
        
        actual = [row[-1] for row in fold]
        # 计算准确率
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores
################# 构造随机森林的一些方法  结束 #################

################# 随机森林测试 开始 #################
# 开始
# 加载数据
filename = 'D:\sonar.all-data.csv'
dataset = load_csv(filename)
# print(dataset)
# 转换数据类型
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)

n_folds = 5 # 把原始数据集分为5份
max_depth = 5
min_size = 10

scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
################# 随机森林测试 结束 #################