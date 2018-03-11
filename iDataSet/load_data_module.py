#!/usr/bin/env python
# -*- coding: utf-8 

from csv import reader

# 加载CSV文件
def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset

filename = 'D:\sonar.all-data.csv'
dataset = load_csv(filename)
print(dataset)
print(len(dataset))