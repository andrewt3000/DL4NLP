### Definition of Machine Learning
A program is said to learn from experience E with respect to some task T & some performance measure P, if it’s performance on T, as measured by P, improves with experience E.  
T - task  
P - probability of improved performance  
E - experience  

### 3 Types of machine learning
1. Supervised learning - learns uses labeled training data.  Example predict house prices in portland.
2. Unsupervised learning - learns using unlabled training data.  algorithm finds structure. Clustering is an example of unsupervised learning. Examples: social network analysis, market segmentation, astronomical data analysis.  
3. Reinforcement learning - learns to take action by maximizing a cumulative reward.

### 2 Types of machine learning problems
1. regression - predicting a continuous value attribute (Example: house prices)
2. classification - predicting a discrete value. Example: benign/malignant tumor, fraudulent or legitimate financial transaction, spam or non-spam email.

###Linear Regression with one variable
m = number of training examples  
x = input features  
y = output variable / target variable  

(x, y) = refers to one training example  
x(i) y(i) refers to specific training example at index i. It doesn’t refer to an exponent.

h = hypothesis. function that maps x to y  
Predict that y is a linear function of x.  
<strong>y = h(x) = ϴ<sub>0</sub> + ϴ<sub>1</sub>x</strong>  

###Cost Function
J is the cost function  
ϴ<sub>i</sub> are parameters. the coefficients of the function.  
find values of ϴ<sub>0</sub> and ϴ<sub>1</sub> that minimize the cost (error) function.  
h(x) - y = difference in function versus actual… we want to minimize this.  
aka squared error cost function.  
J(ϴ<sub>0</sub>, ϴ<sub>1</sub>) = 1/2m sum from i = 1 to m (h(x<sup>(i)</sup>) - y<sup>(i)</sup>)<sup>2</sup>  
find ϴ<sub>0</sub> and ϴ<sub>1</sub> that minimizes the error.  

Square the error because  
1) Squared gets rid of negative numbers that would cancel each other out.  Although you could use magnitude.  
2) For many applications small errors are not important, but big errors are very important.  example: ½ foot steering error no big deal, 5 foot error huge problem. So it’s not 10x more important… it is 100x more important.  


###Cost Function Intuition
simplify and make ϴ<sub>0</sub> = 0
