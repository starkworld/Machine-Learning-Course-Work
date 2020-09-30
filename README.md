# Machine-Learning-Course-Work
Consist of ML and data science code templates and course work
These are the assignment and coursework which I have done while I was taking Machine Learning course online.
### This Repo includes
* Data Preprocessing
### SuperVised Learning Methods
#### Regression models
#### Linear Regression
* Linear Regression is a machine learning algorithm based on supervised learning.
It performs a regression task. Regression models a target prediction value based on independent variables.
It is mostly used for finding out the relationship between variables and forecasting.
Different regression models differ based on – the kind of relationship between dependent
and independent variables, they are considering and the number of independent variables being used.
A simple LR Graph plotting shown belwoe
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/Simple%20Linear%20Regression/Screen%20Shot%202020-09-24%20at%201.40.03%20AM.png)
** Hypothesis function for Linear Regression :
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/Simple%20Linear%20Regression/Screen%20Shot%202020-09-24%20at%201.40.15%20AM.png)
* While training the model we are given :
* x: input training data (univariate – one input variable(parameter))
* y: labels to data (supervised learning)

* When training the model – it fits the best line to predict the value of y for a given value of x. The model gets the best regression fit line by finding the best θ1 and θ2 values.
* θ1: intercept
* θ2: coefficient of x

#### Polynomial Regression
* Polynomial Regression is a regression algorithm that models the relationship between a dependent(y) and independent variable(x) as nth degree polynomial. The Polynomial Regression equation is given below

* Polynomial Regression
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/Polynomial%20LR/Screen%20Shot%202020-09-24%20at%201.50.10%20AM.png)

* y= b0+b1x1+ b2x12+ b2x13+...... bnx1n

* x: input training data (univariate – one input variable(parameter))
* y: labels to data (supervised learning)
* bn: coefficient variable of x

#### Multiple Regression
* Multiple linear regression (MLR), also known simply as multiple regression, is a statistical technique that uses several explanatory variables to predict the outcome of a response variable. The goal of multiple linear regression (MLR) is to model linear relationship between exploratory vatiable and response variable.
* Formula and Calcualtion of Multiple Linear Regression
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/Multiple%20Linear%20Regression/Screen%20Shot%202020-09-25%20at%205.25.12%20PM.png)

#### Support Vector Machine Regression
* Support Vector Machines (SVMs) are well known in classification problems. The use of SVMs in regression is not as well documented, however. These types of models are known as Support Vector Regression (SVR).
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/SVR/Screen%20Shot%202020-09-25%20at%205.30.29%20PM.png)
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/SVR/Screen%20Shot%202020-09-25%20at%205.30.39%20PM.png)
* SVR is a powerful algorithm that allows us to choose how tolerant we are of errors, both through an acceptable error margin(ϵ) and through tuning our tolerance of falling outside that acceptable error rate. Hopefully, this tutorial has shown you the ins-and-outs of SVR and has left you confident enough to add it to your modeling arsenal.

#### Decison Tree Regression
* Decision tree builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. A decision node (e.g., Outlook) has two or more branches (e.g., Sunny, Overcast and Rainy), each representing values for the attribute tested. Leaf node (e.g., Hours Played) represents a decision on the numerical target. The topmost decision node in a tree which corresponds to the best predictor called root node. Decision trees can handle both categorical and numerical data. 
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/Decision%20Trees/Screen%20Shot%202020-09-25%20at%205.38.16%20PM.png)

![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/Decision%20Trees/Screen%20Shot%202020-09-25%20at%205.39.53%20PM.png)

#### Random Forest Regression
* Random forest is a Supervised Learning algorithm which uses ensemble learning method for classification and regression. Random forest is a bagging technique and not a boosting technique. The trees in random forests are run in parallel. There is no interaction between these trees while building the trees.
* It operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/Random%20Forest/Screen%20Shot%202020-09-25%20at%205.41.40%20PM.png)

![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/Random%20Forest/Screen%20Shot%202020-09-25%20at%205.44.33%20PM.png)

#### Classinfication models
#### Logistic Regression
* Logistic regression is a classification algorithm used to assign observations to a discrete set of classes. Some of the examples of classification problems are Email spam or not spam, Online transactions Fraud or not Fraud, Tumor Malignant or Benign. Logistic regression transforms its output using the logistic sigmoid function to return a probability value.
* Types of logistic regression
* Binary (eg. Tumor Malignant or Benign)
* Multi-linear functions failsClass (eg. Cats, dogs or Sheep's)
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Classification%20models/images/Screen%20Shot%202020-09-26%20at%208.27.55%20PM.png)

* The hypothesis of logistic regression tends it to limit the cost function between 0 and 1. Therefore linear functions fail to represent it as it can have a value greater than 1 or less than 0 which is not possible as per the hypothesis of logistic regression.
* Here below you can see the perofrmance of logistics regression on increasing number of epochs
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Classification%20models/images/1*PQ8tdohapfm-YHlrRIRuOA.gif)

* Here you can see the prediction difference between linera regression model and logistic regression model
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Classification%20models/images/Screen%20Shot%202020-09-26%20at%208.27.30%20PM.png)

#### Support Vector Machine Classification
* Support Vector Machines (SVMs) are well known in classification problems. The use of SVMs in regression is not as well documented. The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Classification%20models/images/Screen%20Shot%202020-09-26%20at%208.49.06%20PM.png)

![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Classification%20models/images/Screen%20Shot%202020-09-26%20at%208.49.17%20PM.png)

* To separate the two classes of data points, there are many possible hyperplanes that could be chosen. Our objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Classification%20models/images/Screen%20Shot%202020-09-26%20at%208.48.52%20PM.png)
#### Naive Bayes
* In machine learning we are often interested in selecting the best hypothesis (h) given data (d).

* In a classification problem, our hypothesis (h) may be the class to assign for a new data instance (d).

* One of the easiest ways of selecting the most probable hypothesis given the data that we have that we can use as our prior knowledge about the problem. Bayes’ Theorem provides a way that we can calculate the probability of a hypothesis given our prior knowledge.

* Bayes’ Theorem is stated as:

**P(h|d) = (P(d|h) * P(h)) / P(d)**

* Where

* P(h|d) is the probability of hypothesis h given the data d. This is called the posterior probability.
* P(d|h) is the probability of data d given that the hypothesis h was true.
* P(h) is the probability of hypothesis h being true (regardless of the data). This is called the prior probability of h.
* P(d) is the probability of the data (regardless of the hypothesis).

#### Random Forest Classification
* In machine learning, the random forest algorithm is also known as the random forest classifier. It is a very popular classification algorithm. One of the most interesting thing about this algorithm is that it can be used as both classification and random forest regression algorithm. The RF algorithm is an algorithm for machine learning, which is a forest. We know the forest consists of a number of trees. The trees being mentioned here are decision trees.  Therefore, the RF algorithm comprises a random collection or a random selection of a forest tree. It is an addition to the decision tree algorithm. So basically, what a RF algorithm does is that it creates a random sample of multiple decision trees and merges them together to obtain a more stable and accurate prediction through cross validation. In general, the more trees in the forest, the more robust would be the prediction and thus higher accuracy.
* Why Use random forest??
Random forest algorithm can be used for both classifications and regression task.
* It provides higher accuracy through cross validation.
* Random forest classifier will handle the missing values and maintain the  accuracy of a large proportion of data.
* If there are more trees, it won’t allow over-fitting trees in the model.
* It has the power to handle a large data set with higher dimensionality
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Classification%20models/images/Screen%20Shot%202020-09-27%20at%201.55.35%20AM.png)

![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Classification%20models/images/Random_forest_diagram_complete.png)

#### Decision Tree Classification
* A Decision Tree is a simple representation for classifying examples. It is a Supervised Machine Learning where the data is continuously split according to a certain parameter.
* Decision Tree consists of : \
Nodes : Test for the value of a certain attribute.\
Edges/ Branch : Correspond to the outcome of a test and connect to the next node or leaf.\
Leaf nodes : Terminal nodes that predict the outcome (represent class labels or class distribution).
![alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Classification%20models/images/Screen%20Shot%202020-09-27%20at%202.02.56%20AM.png)
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Classification%20models/images/Screen%20Shot%202020-09-27%20at%202.04.12%20AM.png)

#### K-Nearest Neighbors
* The k-nearest neighbors (KNN) algorithm is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems
* The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.
##### The KNN Algorithm
1.Load the data\
2. Initialize K to your chosen number of neighbors\
3. For each example in the data\
3.1 Calculate the distance between the query example and the current example from the data.\
3.2 Add the distance and the index of the example to an ordered collection\
4. Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances\
5. Pick the first K entries from the sorted collection\
6. Get the labels of the selected K entries\
7. If regression, return the mean of the K labels\
8. If classification, return the mode of the K labels\
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Classification%20models/images/Screen%20Shot%202020-09-27%20at%202.07.52%20AM.png)
**Choosing the right value for K**
* To select the K that’s right for your data, we run the KNN algorithm several times with different values of K and choose the K that reduces the number of errors we encounter while maintaining the algorithm’s ability to accurately make predictions when it’s given data it hasn’t seen before.

### Unsupervised Learning Methods
* No labels are given to the learning algorithm, leaving it on its own to find structure in its input. Unsupervised learning can be a goal in itself (discovering hidden patterns in data) or a means towards an end (feature learning).
* In some pattern recognition problems, the training data consists of a set of input vectors x without any corresponding target values. The goal in such unsupervised learning problems may be to discover groups of similar examples within the data, where it is called clustering, or to determine how the data is distributed in the space, known as density estimation. To put forward in simpler terms, for a n-sampled space x1 to xn, true class labels are not provided for each sample, hence known as learning without teacher.
#### Clustering Models
* Clustering can be considered the most important unsupervised learning problem; so, as every other problem of this kind, it deals with finding a structure in a collection of unlabeled data. A loose definition of clustering could be “the process of organizing objects into groups whose members are similar in some way”. A cluster is therefore a collection of objects which are “similar” between them and are “dissimilar” to the objects belonging to other clusters.
##### K means Clustering
K-means is one of the simplest unsupervised learning algorithms that solves the well known clustering problem. The procedure follows a simple and easy way to classify a given data set through a certain number of clusters (assume k clusters) fixed a priori. The main idea is to define k centres, one for each cluster. These centroids should be placed in a smart way because of different location causes different result. So, the better choice is to place them as much as possible far away from each other. The next step is to take each point belonging to a given data set and associate it to the nearest centroid. When no point is pending, the first step is completed and an early groupage is done. At this point we need to re-calculate k new centroids as barycenters of the clusters resulting from the previous step. After we have these k new centroids, a new binding has to be done between the same data set points and the nearest new centroid. A loop has been generated. As a result of this loop we may notice that the k centroids change their location step by step until no more changes are done. In other words centroids do not move any more.
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Classification%20models/images/Screen%20Shot%202020-09-29%20at%206.49.18%20PM.png)
Finally, this algorithm aims at minimizing an objective function, in this case a squared error function. The objective function.
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Classification%20models/images/Screen%20Shot%202020-09-29%20at%206.50.08%20PM.png)

is a chosen distance measure between a data point xi and the cluster centre cj, is an indicator of the distance of the n data points from their respective cluster centres.
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Classification%20models/images/Screen%20Shot%202020-09-29%20at%206.51.03%20PM.png)

where, ‘ci’ represents the number of data points in ith cluster.
Recalculate the distance between each data point and new obtained cluster centers.
If no data point was reassigned then stop, otherwise repeat from step 3).
##### Hierarchical Clustering
Given a set of N items to be clustered, and an N*N distance (or similarity) matrix, the basic process of hierarchical clustering is this:
Start by assigning each item to a cluster, so that if you have N items, you now have N clusters, each containing just one item. Let the distances (similarities) between the clusters the same as the distances (similarities) between the items they contain.
Find the closest (most similar) pair of clusters and merge them into a single cluster, so that now you have one cluster less.
Compute distances (similarities) between the new cluster and each of the old clusters.
Repeat steps 2 and 3 until all items are clustered into a single cluster of size N.
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Classification%20models/images/Screen%20Shot%202020-09-29%20at%206.52.31%20PM.png)

### Reinforcement learning
* A system interacts with a dynamic environment in which it must perform a certain goal (such as driving a vehicle or playing a game against an opponent). The system is provided feedback in terms of rewards and punishments as it navigates its problem space.
##### Association Rule based Learning
* Association Rules is one of the very important concepts of machine learning being used in market basket analysis. In a store, all vegetables are placed in the same aisle, all dairy items are placed together and cosmetics form another set of such groups. Investing time and resources on deliberate product placements like this not only reduces a customer’s shopping time, but also reminds the customer of what relevant items (s)he might be interested in buying, thus helping stores cross-sell in the process. Association rules help uncover all such relationships between items from huge databases. One important thing to note is - Rules do not extract an individual’s preference, rather find relationships between set of elements of every distinct transaction. This is what makes them different from collaborative filtering.
* Lets now see what an association rule exactly looks like. It consists of an antecedent and a consequent, both of which are a list of items. Note that implication here is co-occurrence and not causality. For a given rule, itemset is the list of all the items in the antecedent and the consequent.
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/reinforcement%20learning/images/Screen%20Shot%202020-09-30%20at%202.51.36%20PM.png)

##### Support
This measure gives an idea of how frequent an itemset is in all the transactions. Consider itemset1 = {bread} and itemset2 = {shampoo}. There will be far more transactions containing bread than those containing shampoo. So as you rightly guessed, itemset1 will generally have a higher support than itemset2. Now consider itemset1 = {bread, butter} and itemset2 = {bread, shampoo}.
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/reinforcement%20learning/images/Screen%20Shot%202020-09-30%20at%202.51.43%20PM.png)
##### Confidence
This measure defines the likeliness of occurrence of consequent on the cart given that the cart already has the antecedents. That is to answer the question — of all the transactions containing say, {Captain Crunch}, how many also had {Milk} on them? We can say by common knowledge that {Captain Crunch} → {Milk} should be a high confidence rule. Technically, confidence is the conditional probability of occurrence of consequent given the antecedent.
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/reinforcement%20learning/images/Screen%20Shot%202020-09-30%20at%202.53.39%20PM.png)

##### Lift
* Lift controls for the support (frequency) of consequent while calculating the conditional probability of occurrence of {Y} given {X}. Lift is a very literal term given to this measure. Think of it as the *lift* that {X} provides to our confidence for having {Y} on the cart. To rephrase, lift is the rise in probability of having {Y} on the cart with the knowledge of {X} being present over the probability of having {Y} on the cart without any knowledge about presence of {X}. Mathematically,
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/reinforcement%20learning/images/Screen%20Shot%202020-09-30%20at%202.54.19%20PM.png)

##### Upper Confidence Bound
This is the essence of a multi-armed bandit problem, which is a simplified reinforcement learning task. These can occur when you’re seeking to maximize engagement on your website, or figuring out clinical trials, or trying to optimize your computer’s performance.\
Bandit problems require a balance between the exploration and exploitation trade-off. Because the problem starts with no prior knowledge of the rewards, it needs to explore (try a lot of slot machines) and then exploit (repeatedly pull the best lever) once it has narrowed down its selections.\
ϵ-greedy can take a long time to settle in on the right one-armed bandit to play because it’s based on a small probability of exploration. The Upper Confidence Bound (UCB) method goes about it differently because we instead make our selections based on how uncertain we are about a given selection.
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/reinforcement%20learning/images/Screen%20Shot%202020-09-30%20at%202.59.57%20PM.png)

The above equation gives us the selection criterion for our model. Q_n(a) is our current estimate of the value of a given slot machine a. The value under the square root is the log of the total number of slot machines we’ve tried, n, divided by the number of times we’ve tried each slot machine a (k_n), while c is just a constant. We choose our next slot machine by selecting whichever bandit gives the largest value at each step n.
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/reinforcement%20learning/images/Screen%20Shot%202020-09-30%20at%203.00.55%20PM.png)

![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/reinforcement%20learning/images/Screen%20Shot%202020-09-30%20at%203.01.11%20PM.png)

##### Thompson Sampling
* Reinforcement Learning is a branch of Machine Learning, also called Online Learning. It is used to decide what action to take at t+1 based on data up to time t. This concept is used in Artificial Intelligence applications such as walking. A popular example of reinforcement learning is a chess engine. Here, the agent decides upon a series of moves depending on the state of the board (the environment), and the reward can be defined as win or lose at the end of the game.
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/reinforcement%20learning/images/Screen%20Shot%202020-09-30%20at%203.06.45%20PM.png)

* Thompson Sampling (Posterior Sampling or Probability Matching) is an algorithm for choosing the actions that address the exploration-exploitation dilemma in multi-armed bandit problem. Actions are performed several times and are called exploration. It uses training information that evaluates the actions taken rather than instructs by giving correct actions. This is what creates the need for active exploration, for an explicit trial-and-error search for good behaviour. Based on the results of those actions, rewards (1) or penalties (0) are given for that action to the machine. Further actions are performed in order to maximize the reward that may improve future performance. Suppose a robot has to pick several cans and put in a container. Each time it puts the can to the container, it will memorize the steps followed and train itself to perform the task with better speed and precision (reward). If the Robot is not able to put the can in the container, it will not memorize that procedure (hence speed and performance will not improve) and will be considered as a penalty.
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/reinforcement%20learning/images/Screen%20Shot%202020-09-30%20at%203.07.30%20PM.png)

*Some Practical Applications* 

* Netflix Item based recommender systems: Images related to movies/shows are shown to users in such a way that they are more likely to watch it.
* Bidding and Stock Exchange: Predicting Stocks based on Current data of stock prizes.
* Traffic Light Control: Predicting the delay in signal.
Automation in Industries: Bots and Machines for transporting and Delivering items without human intervention.

References:
https://towardsdatascience.com/
https://www.geeksforgeeks.org/introduction-to-thompson-sampling-reinforcement-learning/
