import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import queue
from Silhouette_Score import Silhouette_score
from Calinski_Harbaz import Calinski_Harabasz_score

import math

def Feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    x_norm = np.subtract(X , mu)
    x_norm = np.divide(x_norm, sigma)

    return x_norm
# differnt point group
core = -1
edge = -2


#function to find all neigbor points in radius
def Eudist(data, num, eps):
    points = []
    for i in range(len(data)):
        if np.linalg.norm(data[i] - data[num]) <= eps:
            points.append(i)
    return points

#DB Scan algorithom
def dbscan(data, eps, minpt):
    #initilize all pointlable to unassign
    pointlabel  = [0] * len(data)
    pointcount = []
    #initilize list for core and noncore point
    corepoint=[]
    noncore=[]
    
    #Find all neigbor for all point
    for i in range(len(data)):
        pointcount.append(Eudist(data,i,eps))
    
    #Find all core point, edgepoint and noise
    for i in range(len(pointcount)):
        if (len(pointcount[i])>=minpt):
            pointlabel[i]=core
            corepoint.append(i)
        else:
            noncore.append(i)

    for i in noncore:
        for j in pointcount[i]:
            if j in corepoint:
                pointlabel[i]=edge
                break
            
    #start assigning point to cluster
    cl = 1

    for i in range(len(pointlabel)):
        q = queue.Queue()
        if (pointlabel[i] == core):
            pointlabel[i] = cl
            for x in pointcount[i]:
                if(pointlabel[x]==core):
                    q.put(x)
                    pointlabel[x]=cl
                elif(pointlabel[x]==edge):
                    pointlabel[x]=cl
            #Stop when all point in Queue has been checked   
            while not q.empty():
                neighbors = pointcount[q.get()]
                for y in neighbors:
                    if (pointlabel[y]==core):
                        pointlabel[y]=cl
                        q.put(y)
                    if (pointlabel[y]==edge):
                        pointlabel[y]=cl            
            cl=cl+1 #move to next cluster      
    return pointlabel,cl
    
def p1plot(data, datalabel,cl):
  colors=['black', 'green', 'brown', 'red', 'purple', 'orange', 'yellow']
  for i in range(cl):
    if i==0:
      co='blue'
    else:
      co=colors[i%len(colors)]
    x1=[]
    x2=[]
    for j in range(len(data)):
      if datalabel[j]==i:
        x1.append(data[j,0])
        x2.append(data[j,1])
    plt.scatter(x1, x2, c=co, alpha=1, marker='.')

if __name__ == '__main__':

    # Set EPS and Minpoint
    #eps = 2
    #minpts = 5

    df = pd.read_excel(r'data.xlsx')
    data = df.to_numpy()

    data = Feature_normalize(data)
    #data = X[:,[2,4]]

    #pointlabel, cl = dbscan(data, eps, minpts)
    #print("Calculted by Algorithm")
    #p1plot(data, pointlabel, cl)
    #plt.show()
    m_cs = -100
    opt_eps_s = -1
    opt_minpts_s = -1

    m_c = -100
    opt_eps_c = -1
    opt_minpts_c = -1
    # loop to find out optimum epsilon and min point
    for eps in np.arange(1.9,3.2,0.1):
        for minpts in range(3,8):
            pointlabel, cl = dbscan(data, eps, minpts)
            pointlabel = np.array(pointlabel )
            pointlabel = pointlabel-1
            #print(pointlabel)
            S_c = Silhouette_score(data, pointlabel)
            cal_h = Calinski_Harabasz_score(data, pointlabel)
            if m_cs<S_c:
                m_cs = S_c
                opt_eps_s = eps
                opt_minpts_s = minpts

            if m_c<cal_h:
                m_c = cal_h
                opt_eps_c = eps
                opt_minpts_c = minpts
            print('Silhouette score for K =',cl-1,'eps =',eps,'minpts = ',minpts, 'is ', S_c)
            print('Calinski Harabasz score for K =',cl-1,'eps =',eps,'minpts = ',minpts, 'is ', cal_h,'\n')


    print('Max Silhouette score for eps =',opt_eps_s, 'and min points = ',opt_minpts_s, 'is',m_cs )
    print('Max Calinski Harabasz score for eps =', opt_eps_c, 'and min points = ', opt_minpts_c, 'is', m_c)







