#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:55:57 2020

@author: srivenkat
"""
import sys
import copy

print('')

if len(sys.argv) != 2:
    print("Invalid number of arguments. Program terminating...")
    sys.exit(0)

filename=sys.argv[1]

try:
    fptr = open(filename,"r")
except:
    print("File not found. Program terminating...")
    sys.exit(0)

file_data = fptr.readlines()
fptr.close()

data_length = len(file_data)

for i in range(data_length):
    word=file_data[i].split()
    if word[0]=='.circuit':
        break

i_start = i
if i_start==data_length-1:
    print("No relevant data in the chosen file. Program terminating...")
    sys.exit(0)

while 1:
    #print(file_data[i])
    i += 1
    word=file_data[i].split()
    if word[0]=='.end':
        break

i_end = i

#print(file_data[i])

netlist = []
element={}

num = i_end-i_start-1

for i in range(i_start+1,i_end):
    data = file_data[i].split()
    
    if data[0]!='R' or data[0]!='L' or data[0]!='C' or data[0]!='V' or data[0]!='A':
        element['name'] = data[0]
        element['node1'] = data[1]
        element['node2'] = data[2]
        element['value'] = data[3]
    elif data[1]!='E' or data[1]!='G' or data[1]!='H' or data[1]!='F':
        element['name'] = data[0]    #print(file_data[i])
        element['node1'] = data[1]
        element['node2'] = data[2]
        element['node3'] = data[3]
        element['node4'] = data[4]
        element['value'] = data[5]
    else:
        print('Invalid data in chosen file. Program terminating...')

    netlist.append(copy.deepcopy(element))
    #print(netlist[i-i_start-1])

print("\nnetlist updated\n")

print("The circuit elements in reverse order are:")

for i in range(num):
    temp_item=list(netlist[num-i-1].values())
    count=len(temp_item)
    for j in range(count):
        print(temp_item[len(netlist[i])-j-1],end=' ')
    print(' ')
