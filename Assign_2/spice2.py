#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 11:23:54 2020
@author: srivenkat
"""
# specify the nature of source ac/dc. If ac, add a separate line for frequency. Assign 1 node in the ckt as GND node

import numpy as np
import sys
import copy
import cmath

CIRCUIT = '.circuit'
END = '.end'
AC = '.ac'
GND = 'GND'
ZERO = 0.000000000000000000000000001    #will operate as frequency in DC circuits
print('')

if len(sys.argv) != 2:  #check if name of file is given as argument
    print("Invalid number of arguments. Program terminating...")
    sys.exit(0)

filename=sys.argv[1]

#check if file exists
try:
    fptr = open(filename,"r")
except:
    print("File not found. Program terminating...")
    sys.exit(0)

file_data = fptr.readlines()    #read file data into a list of lines
fptr.close()

data_length = len(file_data)

node_list = []   #list of all unique nodes
comp = []        #list of objects for all components(R,L,C)
source = []      #list of objects for all sources I/V and AC/DC
Volt = []        #list of all V sources AC/DC
Amp = []         #list of all I sources AC/DC

#define specific classes for passive elements and sources
class Component: 
    name =' '   #R #L #C
    value = 0
    node_a = -1     #terminal nodes of the component
    node_b = -2     #terminal nodes of the component

class Source:
    name =' '   #I #V
    value = 0
    node_a = -1     #terminal nodes of the component
    node_b = -2     #terminal nodes of the component
    div = ' '       #ac or dc nature of source
    phase = 0
    freq = ZERO
    offset = 0

#extract data from file
for i in range(data_length):
    line=file_data[i].split()   #temperorily store the split lines
    if line[0]==CIRCUIT:    #index of first line of netlist '.circuit'
        break
    
i_start = i     #index of '.circuit'

if i_start==data_length-1:
    print("No relevant data in the chosen file. Program terminating...")
    sys.exit(0)

while 1:
    i += 1
    line=file_data[i].split()   #temperorily store the split lines
    if line[0]==END:    #index of last line of netlist 'end'
        break

i_end = i       #index of '.end'

v_count = 0     #no. of independent voltage sources
i_count = 0     #no. of independent current sources
comp_count = 0  #no. of passive elements
source_count = 0
node_count = 0

#extract data from file into corresponding objects and lists
for i in range(i_start+1,i_end):
    line = file_data[i].split()
    if line[0][0]=='R' or line[0][0]=='L' or line[0][0]=='C':
        comp.append(Component())    #storing R,L,C objects in a separate list
        comp[comp_count].name = line[0]
        comp[comp_count].node_a = line[1]
        comp[comp_count].node_b = line[2]
        comp[comp_count].value = float(line[3])
        comp_count += 1
    elif line[0][0]=='V' or line[0][0]=='I':
        if line[3] == 'dc' or line[3] == 'ac':
            source.append(Source())
            source[source_count].name = line[0]
            source[source_count].node_a = line[1]
            source[source_count].node_b = line[2]
            source[source_count].div = line[3]
            source[source_count].value = float(line[4])
            try:    #check for comments (default phase and offest is 0)
                source[source_count].phase = line[5] if line[5][0] != '#' else 0
                if line[5][6] != '#':
                    source[source_count].offset = line[6] 
                else: 
                    source[source_count].offset = 0
            except:
                pass
            if line[0][0] == 'V':
                Volt.append(source[source_count])
                v_count += 1
            else:
                Amp.append(source[source_count])
                i_count += 1
            source_count += 1
        else:
            print("Unspecified nature of source. Program terminating...")
            sys.exit(0)
    else:
        print("Unknown circuit element found. Program terminating...")
        sys.exit(0)

real = 0    #arguments to 'complex' function
imag = 0

#extracting and setting frequency and phase for all AC elements
frequency = ZERO
for i in range(i_end, data_length):
    line = file_data[i].split()
    if line[0] == AC:   #check for '.ac' line
        frequency = line[2]
        frequency = 2.00*cmath.pi*float(frequency)      #converting Hz to rad/sec

for i in range(len(source)):
    node_list.append(source[i].node_a)
    node_list.append(source[i].node_b)
    if source[i].div == 'ac':
        source[i].freq = frequency
        source[i].value /= 2  #peak to peak value is given. div / 2 to get max. amplitude
    elif source[i].div == 'dc' and frequency != ZERO:
        print("Circuit cannot operate at 2 different frequencies. Program terminating...")
        sys.exit(0)
    phase = complex(0,float(source[i].phase))
    phi = cmath.exp(phase)
    source[i].value = complex(float(source[i].value),0) + complex(float(source[i].offset))
    source[i].value *= phi

#updating node list and complex impedances
for i in range(len(comp)):
    node_list.append(comp[i].node_a)
    node_list.append(comp[i].node_b)
    if comp[i].name[0] == 'R':
        pass
    elif comp[i].name[0] == 'L':
        imag = frequency * comp[i].value        #Z = (jwL)
        comp[i].value = complex(real,imag)
    else:   #Capacitor element
        imag = -1/(frequency * comp[i].value)      #Z = 1/(jwc)
        comp[i].value = complex(real,imag)

#print('netlist updated')

node_list = np.unique(node_list)     #extracting all the unique nodes in the circuit
node_count = len(node_list)     #no. of KCL equations

#check if GND node is specified
if GND not in node_list:
    print('No specified GND node found. Program terminating...')
    sys.exit(0)
gnd_node = -1

mat_num = node_count + v_count      #no. of rows in matrix M = no. of columns = nodes + voltage sources 

#predefining zero matrices for MX=B
M_mat = np.zeros(shape=[mat_num,mat_num],dtype='complex')   #assigning 0+0j to all elements to selectively replace the values as per KCL equations
X_mat = np.zeros(shape=[mat_num,1],dtype='complex')
B_mat = np.zeros(shape=[mat_num,1],dtype='complex')

# first 'node_count' rows in X_mat are for the node voltages, in the same order as in node_list.
# next 'v_count' rows are the current values through the known voltage sources
v_row = 0
for j in range(source_count):
    element2 = copy.deepcopy(source[j])    #temperorily storing necessary object in element
    node1 = element2.node_a
    node2 = element2.node_b
    node1 = list(node_list).index(node1)
    node2 = list(node_list).index(node2)
    if element2.name[0] == 'V':
        M_mat[node1][node_count+v_row] = -1      #unknown current variable enters node1
        M_mat[node2][node_count+v_row] = 1      #unknown current variable exits node2
        M_mat[node_count+v_row][node1] = 1
        M_mat[node_count+v_row][node2] = -1
        B_mat[node_count+v_row] = element2.value
        v_row += 1
    elif element2.name[0] == 'I':
        B_mat[node1] += element2.value
        B_mat[node2] -= element2.value
    else:
        print("Unknown source element found. Program terminating...")
        sys.exit(0)

#framing KCL equations into M_mat
for i in range(node_count):
    row_temp = [0]*mat_num
    element = ()
    if node_list[i] == GND:
        gnd_node = i
        row_temp[i] = 1
        M_mat[i] = np.array(row_temp)
        B_mat[i] = 0
        continue    #separate row needed for GND node. 
    for j in range(comp_count):
        if comp[j].node_a == node_list[i] or comp[j].node_b == node_list[i]:
            element = copy.deepcopy(comp[j])    #temperorily storing necessary object in element
            node1 = node_list[i]
            node2 = element.node_b if element.node_a == node_list[i] else element.node_a
            node1 = list(node_list).index(node1)
            node2 = list(node_list).index(node2)
            row_temp[node1] += 1/element.value
            row_temp[node2] += -1/element.value
    row_temp = np.array(row_temp)
    M_mat[i] += row_temp

try:    #checking for singular matrix M_mat, caused due to netlist error
    X_mat = np.linalg.solve(M_mat,B_mat)
except:
    print("Netlist error. Circuit cannot be solved. Program terminating...")
    sys.exit(0)

#rounding off solution of Voltages and Currents to 6 digits of precision
x_rows,x_cols = np.shape(X_mat)
for i in range(x_rows):
    z = X_mat[i]
    X_mat[i] = complex(np.round(z.real,6),np.round(z.imag,6))
''' 
#to print the matrices in MX=B
print("\n M matrix equals:\n",M_mat)
print("\n B matrix equals:\n",B_mat)
print("\n X matrix equals\n",X_mat)
'''
#print solutions for all the unknown node voltages and currents through voltage sources
print('node voltages: ')
for i in range(node_count):
    if frequency == ZERO:
        print(node_list[i],':',X_mat[i].real,end='    ')
    else:
        print(node_list[i],':',X_mat[i],end='    ')
print('\n\ncurrent through voltage sources: ')
for i in range(v_count):
    if frequency == ZERO:
        print(Volt[i].name,':',X_mat[i+node_count].real,end='    ')
    else:
        print(Volt[i].name,':',X_mat[i+node_count],end='    ')




