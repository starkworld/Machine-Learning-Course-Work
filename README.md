# Machine-Learning-Course-Work
Consist of ML and data science code templates and course work
These are the assignment and coursework which I have done while I was taking Machine Learning course online.
### This Repo includes
* Data Preprocessing
### SuperVised Learning Methods
#### Regression models
##### Linear Regression
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

##### Polynomial Regression
* Polynomial Regression is a regression algorithm that models the relationship between a dependent(y) and independent variable(x) as nth degree polynomial. The Polynomial Regression equation is given below

* Polynomial Regression
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/Polynomial%20LR/Screen%20Shot%202020-09-24%20at%201.50.10%20AM.png)

* y= b0+b1x1+ b2x12+ b2x13+...... bnx1n

* x: input training data (univariate – one input variable(parameter))
* y: labels to data (supervised learning)
* bn: coefficient variable of x

##### Multiple Regression
* Multiple linear regression (MLR), also known simply as multiple regression, is a statistical technique that uses several explanatory variables to predict the outcome of a response variable. The goal of multiple linear regression (MLR) is to model linear relationship between exploratory vatiable and response variable.
* Formula and Calcualtion of Multiple Linear Regression
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/Multiple%20Linear%20Regression/Screen%20Shot%202020-09-25%20at%205.25.12%20PM.png)

##### Support Vector Machine Regression
* Support Vector Machines (SVMs) are well known in classification problems. The use of SVMs in regression is not as well documented, however. These types of models are known as Support Vector Regression (SVR).
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/SVR/Screen%20Shot%202020-09-25%20at%205.30.29%20PM.png)
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/SVR/Screen%20Shot%202020-09-25%20at%205.30.39%20PM.png)
* SVR is a powerful algorithm that allows us to choose how tolerant we are of errors, both through an acceptable error margin(ϵ) and through tuning our tolerance of falling outside that acceptable error rate. Hopefully, this tutorial has shown you the ins-and-outs of SVR and has left you confident enough to add it to your modeling arsenal.

##### Decison Tree Regression
* Decision tree builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. A decision node (e.g., Outlook) has two or more branches (e.g., Sunny, Overcast and Rainy), each representing values for the attribute tested. Leaf node (e.g., Hours Played) represents a decision on the numerical target. The topmost decision node in a tree which corresponds to the best predictor called root node. Decision trees can handle both categorical and numerical data. 
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/Decision%20Trees/Screen%20Shot%202020-09-25%20at%205.38.16%20PM.png)

![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/Decision%20Trees/Screen%20Shot%202020-09-25%20at%205.39.53%20PM.png)

##### Random Forest Regression
* Random forest is a Supervised Learning algorithm which uses ensemble learning method for classification and regression. Random forest is a bagging technique and not a boosting technique. The trees in random forests are run in parallel. There is no interaction between these trees while building the trees.
* It operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/Random%20Forest/Screen%20Shot%202020-09-25%20at%205.41.40%20PM.png)

![Alt text](https://github.com/starkworld/Machine-Learning-Course-Work/blob/master/Regression%20Models/Random%20Forest/Screen%20Shot%202020-09-25%20at%205.44.33%20PM.png)

#### Classinfication models
* Logistic Regression
* Support Vector Machine Classification
* Naive Bayes
* Random Forest Classification
* Kernal Support Vector Machine
* Decision Tree Classification
* K-Nearest Neighbors
### Unsupervised Learning Methods
#### Clustering Models
* K means Clustering
* Hierarchical Clustering
### Reinforcement learning(Little)
* Association Rule based Learning
* Upper Confidence Bound
* Thompson Sampling

This Reo consist assignments of every model that written above.
