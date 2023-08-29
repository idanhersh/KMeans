# KMeans
KMeans Algorithm for clustering records of a dataset using Apache Spark
I build a class named Kmeans, that uses the Kmeans Algorithm to cluster similar records of a dataframe.

I built several helper functions (methods of the class):

1. A method named "closetcenter" that finds the closest center to the tested point.

2. A method named "reduce" that maps the points sent to it to a new centroid based on the average of the points.

3. A method named "distcalc" that calculates the Euclidean distance between two points in an n-dimensional space. This method was used to check the distance of each point from the centroid.

4. A method named "criteria_reach" that returns a boolean value of TRUE if the new centroids have changed from the existing centroids according to the pre-defined threshold. If one centroid meets the function's condition, it returns TRUE.

5. A method named "run" that runs the main "mapreduce" function for EXP iterations, adds the results of the function to lists named CH and ARI, and finally returns the average and standard deviation of the CH and ARI values for all runs.

