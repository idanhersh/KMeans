# KMeans
KMeans Algorithm for clustering records of a dataset using Apache Spark.
I built a class named Kmeans, that uses the Kmeans Algorithm to cluster similar records of a dataframe. Using 
We have constructed a class named "Kmeans" that contains its own fields where we store the relevant information for our calculations. These include structures for holding the data such as a list for the true class of the data, the filename, the dataframe, the value of K, the threshold, the maximum number of iterations, the centroids from the previous iteration, the new centroids, and the allocation of points to clusters.

Furthermore, we created a method named "newdata" which takes a link to a CSV file as input, loads the file, removes the header, updates the relevant field with the true class list of the data, and then removes this column. Following this, we perform casting of the data from string to float type and standardize the data.
I built several helper functions (methods of the class):

1. A method named "closetcenter" that finds the closest center to the tested point.

2. A method named "reduce" that maps the points sent to it to a new centroid based on the average of the points.

3. A method named "distcalc" that calculates the Euclidean distance between two points in an n-dimensional space. This method was used to check the distance of each point from the centroid.

4. A method named "criteria_reach" that returns a boolean value of TRUE if the new centroids have changed from the existing centroids according to the pre-defined threshold. If one centroid meets the function's condition, it returns TRUE.

5. A method named "run" that runs the main "mapreduce" function for EXP iterations, adds the results of the function to lists named CH and ARI, and finally returns the average and standard deviation of the CH and ARI values for all runs.

A method named "mapreduce" serves as the main function in the class. Initially, the function creates K centers and stores them in the appropriate field of the class. Then, the function runs in a while loop for a predefined number of iterations, thus limiting the algorithm's execution to I iterations, given that the desired threshold has not been achieved.

Afterward, the function employs the map operation to assign each point to one of the generated centers, creating a list where each element contains the new center and the point mapped to it. The code line for this operation is: `l_points = list(map(self.closestcenter, self.df.collect()))`.

Additionally, the function performs the combine operation by creating a local dictionary that holds keys as the values of the centers to which the points are mapped, and values as lists of all the points mapped to the same center.

The function generates the new centers with the points mapped to them using the "reduce" method. It does this by iterating over the values stored in the local dictionary, converting them into a dedicated list (two code lines: `c_dict = list(map(list, centroid_dict.items()))` and `self.kcenters = [self.reduce(p[1]) for p in c_dict]`), thereby completing the reduce operation.

The function updates the old and new centroids in their corresponding fields in the class. The function checks if we've reached the last iteration each time using the "criteria_reach" method. In the final step, the function creates a list of the class classification based on the original order of the points in the dataset. It prepares the data for the two evaluation tests and returns a list with two elements: the Calinski and Harabasz score and the Rand index adjusted for chance.

