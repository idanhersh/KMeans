# Databricks notebook source
import random 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pyspark
from pyspark.sql import SparkSession,DataFrame,SQLContext
from pyspark.sql.types import *
import pandas as pd
import pyspark.pandas as ps
import functools
from pyspark.sql.functions import stddev, mean, col,array_contains
import sklearn
from sklearn.metrics import adjusted_rand_score,calinski_harabasz_score






class Kmeans():
    
    def __init__(self,filename,k,CT=0.0001,I=30,Exp=10):
        self.filename=filename
        self.df=self.newdata(self.filename) #inputed data
        self.exp=Exp
        self.k=k
        self.ct=CT    # threshold
        self.i=I      # number of iteration allowed
        self.oldcenters=[]
        self.kcenters=[]
        self.values = []
        
        #standardize
    def newdata(self,filename): #turn csv to df
        df1 = spark.read.format("csv").option("header", "true").load(filename)
        df2=sc.textFile(filename,2).map(lambda x: x.split(','))
        scv_df = df2
        pd_df=df1.to_pandas_on_spark() 
        schema=list(pd_df.columns.values)
        df=sc.textFile(filename,2).map(lambda x: x.split(','))
        drop=df.mapPartitionsWithIndex(lambda id_x,iter:list(iter)[8:] if (id_x ==0) else iter)
        drop.collect()
        out=spark.createDataFrame(drop)
        outf=out.select(out.columns[:-1])
        self.origclass=out.select(out.columns[-1])
        self.origclass=list(self.origclass.withColumn("_"+str(len(schema)),self.origclass["_"+str(len(schema))].cast(FloatType())).collect())
        self.origclass=[i[0] for i in self.origclass]
        for i in range(1,len(schema)):
            outf=outf.withColumn("_"+str(i),outf["_"+str(i)].cast(FloatType()))
            meant, sttdevt = outf.select(mean("_"+str(i)), stddev("_"+str(i))).first()
            outf=outf.withColumn("_"+str(i), (col("_"+str(i)) - meant) / sttdevt)
        return outf

    def mapreduce(self):
        self.kcenters=self.df.rdd.takeSample(False,self.k) #choose k random data points
        iterations=0
        while iterations <= self.i: #update centroids kmeans algorithm 
          #  map part
          l_points=list(map(self.closestcenter,self.df.collect()))
          #groupby centroid 
          centroid_dict = {}  
          for i in range(len(l_points)):
            if tuple(l_points[i][0]) not in centroid_dict.keys():
              centroid_dict[tuple(l_points[i][0])] = [l_points[i][1]]
            else:
              centroid_dict[tuple(l_points[i][0])].append(l_points[i][1])  
          # reduce part
          self.oldcenters= self.kcenters
          c_dict=list(map(list, centroid_dict.items()))
          self.kcenters= [self.reduce(p[1]) for p in c_dict] #reduction to new centroids
            
          if iterations > 0:
              #check if threshold criteria
              if not self.criteria_reach(self.kcenters,self.ct):
                  break
          
          self.values = list(map(list, centroid_dict.values()))
        
          
          iterations+=1
        # after break
#         ch = (self.df.collect(), self.values)
        milon ={}
        for i in range(len(self.values)):
            for j in range(len(self.values[i])):
                milon[self.values[i][j]] = i
        predictedclass=[milon[i] for i in self.df.collect()]
        ar=sklearn.metrics.adjusted_rand_score(self.origclass,predictedclass) #adjusted rand score calculation
        ch=sklearn.metrics.calinski_harabasz_score(self.df.collect(),predictedclass)#calinski harabasz score calculation
        return [ch,ar]

    def closestcenter(self,point): #finds closest center to point
        
        closest_point=[self.kcenters[0],self.distcalc(point,self.kcenters[0])]#define first kcenter as closest until others are checked
        for j in range(1,len(self.kcenters)):
            if self.distcalc(point,self.kcenters[j]) < closest_point[1]:
              closest_point=[self.kcenters[j],self.distcalc(point,self.kcenters[j])] #update closest center to that point
        return [closest_point[0], point] #returns key(nearest centroid):value(point)

      # calculating new centroid
    def reduce(self, points):  # point is [closest points list], each closest point is a vector of x1,x2,x3,..
        d = len(points[0])
        new_centroid = [0] * d
        for p in points:
          for i in range(d):
            new_centroid[i] += p[i]
        for i in range(len(new_centroid)):
          new_centroid[i] = new_centroid[i] / len(points)
        return new_centroid
        
    def distcalc(self,point,centroid): #calculate dist of point from a centroid
        sum=0
        v1,v2=point,centroid
        for i in range(len(v1)):
          sum += (v1[i]-v2[i])**2
        return np.sqrt(sum)

    def criteria_reach(self,new_centroids,threshold): #method checks threshold
        for i in range(len(new_centroids)):
          if self.distcalc(new_centroids[i], self.oldcenters[i]) >= self. ct:
            return True
        return False        #all new centroids are less than threshold distance away

    def run(self):
        ch,ari=[],[] #list of all 10 iterations score
        for i in range(self.exp):
          answer = self.mapreduce()
          ch.append(answer[0])
          ari.append(answer[1])
        return [["k="+str(self.k)],["calinski_harabasz:","mean="+str(np.mean(ch)),"std="+str(np.std(ch))],["adjusted_rand:","mean="+str(np.mean(ari)),"std="+str(np.std(ari))]]

kvalues=[2,3,4,5,8]
files=["iris.csv","glass.csv","parkinsons.csv"]
for i in files:
    print(i)
    for j in kvalues:
        k1=Kmeans("dbfs:/FileStore/shared_uploads/hershkoi@post.bgu.ac.il/"+i,j) #k=4, iris dataset
        print(k1.run())








# COMMAND ----------


