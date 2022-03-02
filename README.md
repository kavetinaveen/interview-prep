# Data Science Interview Preparation

## Optimization

**Objective:** Minimization of cost function or loss function

**Optimization process:** Update parameters iteratively to reach local or global minima of cost function

---

**Batch Gradient Descent or Gradient Descent:** Update parameters in the opposite direction of the gradient of the cost function by considering whole data at once

					theta = theta - lr * gradient(cost function w.r.t theta given whole data)

- Where `lr` is the hyper-parameter for learning rate
- Number of records = 100; Number of epochs = 10 => Number of parameter updates = 10

**Cons:**

- If the training data is too big then it's difficult to fit the entire data into memory for gradient computation
- Redundent gradient computation for similar examples
- Online learning is not possible as parameters updates with whole data

---

**Stochastic Gradient:** Update parameters in the opposite direction of the gradient of the cost function, consider one record at a time

					theta = theta - lr * gradient(cost function w.r.t theta given one record of the data)

- Where `lr` is the hyper-parameter for learning rate
- Number of records = 100; Number of epochs = 10 => Number of parameter updates = 1000

**Pros:**

- Main advantage with SGD is, we can update parameters online for new observations

**Cons:**

- Oscillates more as the number of updates are more, however decreasing the learning rate slowly over iteration SGD converges to local or global minima

---

**Mini-Batch Gradient Descent:** Update parameters in the opposite direction of the gradient of the cost function, consider one batch of records at a time

					theta = theta - lr * gradient(cost function w.r.t theta given one batch records of the data)

- Where `lr` is the hyper-parameter for learning rate
- Number of records = 100; Batch size = 10; Number of epochs = 10 => Number of parameter updates = 100

**Pros:**

- It's a trade-off between GD and SGD, reduces oscillations as we consider batch of records not one record at a time

**Cons:**

- Choosing optimal learning rate is difficult, very small lr takes a lot of time to converge; too high lr oscillates around local or global minima or even diverge
- Assumes same learning rate for all features of the data, if the data is sparse and features have very different frequencies then we might not want to update all the parameters at same extent
- Using annealing process we can reduce the learning rate over the iterations either predefince schedule or depending on the change in the objective between epochs falls below a threshold. Either of these hyper parameters are pre-defined by the user and unable to adapt to dataset's characteristics
- Stucks at suboptimal local minima or at saddle point

---

**Momentum:** SGD moves in the opposite direction of the gradient of the cost function, SGD moves too solw in the areas where the surface curves much more steeply in one dimension than in another. Reason for this is, too low learning rate will not allow to takes larger steps even in steepy curves and increasing learning rate may diverge. One solution for this could be accelarate SGD in the relevant diection and slow down when it goes to the opposite direction. This is possible with adding a momentum term.

					Vt = gamma * Vt-1 + lr * gradient(cost function w.r.t theta)
					theta = theta - Vt

- Where gamma is a hyperparameter and recommended value for gamma is 0.9

**Intuition behind this approach:** If we are moving in the same direction as previous step then we add 90% value of previos update value to the current update to accelrate the movement. When we reach near local optima the sign of the gradient changes and thus it pulls back the parameter values to reach local minima.

----

**Nesterov accelerated gradient:** It adds some intelligence to momentum updates. Idea behind this approach is that, we already know partial future position of parameter value and why can't we compute gradient from this position instead of current position. We know that, parameter update values for the next step is "Theta - gamma * Vt-1 - lr * gradient(cost function w.r.t theta)". Since we already know the partial position "Theta - gamma * Vt-1" we compute gradient at this position instead of current position.

					Vt = gamma * Vt-1 + lr * gradient(cost function w.r.t theta - gamma * Vt-1)
					theta = theta - Vt

**Intuition behind this approach:** Momentum simmulates the behaviour of a balls rolling down the hill, whereas NAG simmulates the behavior of an intelligent ball which can see the near future and based on the slope of near future it takes action whether to slowdown or accelarate

---

**Adagrad:** All of the approaches discussed above uses same learning rate for all parameter updates. Adagrad adapts the learning rate of the parameters. Performs smaller updates for parameters associated with frequently occuring features, performs larger updates for parameters associated with infrequent features. It's a well-suited algorithm for the sparse data.

					theta = theta - lr * (1/sqrt(Gt + epsilon)) * gradient(cost function w.r.t theta)

- Where, Gt is a diagonal matrix with ith diagonal element equals to the sum of squares of past gradients of ith parameter

**Intuition behind this approach:** Gradient w.r.t parameter associated with sparse feature is too low; gradient w.r.t parameter associated with frequent feature is too high. Learning rate of the parameter is inversely proportional to it's past gradients, if past gradients are too small (sparse) then increase the learning rate and if past gradients are too high (frequent) then decrease the learning rate.

**Cons:** The major drawback of Adagrad approach is that, Gt -> inf as iterations increasing and thus learning rate -> 0 which means no update to the parameter value

---

**Adadelta:** To overcome from the drawback of adagrad, adadelta considers decaying average of all past squared gradients instead of sum of squares of past gradients.

					E[g^2] at t = gamma * (E[g^2] at t-1) + (1 - gamma) * g^2 at t
					Theta = Theta - lr * (1/sqrt(E[g^2] at t + epsilon)) * gradient(cost function w.r.t theta)


---

**RMSProp:** Let gamma = 0.9 in Adadelta then it becomes RMSProp

					E[g^2] at t = 0.9 * (E[g^2] at t-1) + (1 - 0.1) * g^2 at t
					Theta = Theta - lr * (1/sqrt(E[g^2] at t + epsilon)) * gradient(cost function w.r.t theta)
					
---

**Adam:** It's a combination of both momentum and adadelta. It computes the decaying average of past gradients (momentum) and also decaying average of past squares of gradients (adadelta) to update the parameters.

					mt = beta1 * mt-1 + (1 - beta1) * gradient(cost function w.r.t theta)
					vt = beta2 * vt-1 + (1 - beta2) * gradient(cost function w.r.t theta)^2

Since, it initializes mt and vt with zeros it's biased towards zeros, especially when beta1 and beta2 are close to 1. So authors of Adam proposed below correction.

					mt_hat = mt/(1 - beta1)
					vt_hat = vt/(1 - beta2)

					theta = theta - lr * (1/sqrt(vt_hat + epsilon)) * mt_hat

---

## Supervised Machine Learning Models

**Linear Regression:** It builds a linear relationship in terms of regression coefficients between one or more exogenouos features and one endogenous feature

- Representation -> Y = X * beta + epsilon

	Where, beta is the regression coefficient; epsilon is the random error term; X is the design matrix; Y is the dependent variable

- Transformation  -> Y^hat = X * beta^hat

	Where Y^hat is the predicted values of Y; beta^hat is the estimated statistics of population parameters beta

- Cost function -> (1/n) * (Y - Y^hat)^t * (Y - Y^hat) -> (1/n) * sum((Y - Y^hat)^2)

	Where n is the number of observations in the training data

- Gradient of the cost function -> X^t * (Y^hat - Y)

- Estimated statistic of regression coefficients -> beta^hat = (X^t * X)^(-1) * X^t * Y

- Transformation -> Y^hat = X * (X^t * X)^(-1) * X^t * Y

	Where, H = X * (X^t * X)^(-1) * X^t is called as hat matrix as it keeps hat to the dependent variable :)

- Goodness of fit measure: Multiple R-squared and Adjusted R-squared

- Multiple R-squared (R^2) = Regression sum of squares/Total sum of squares = sum(Y^hat - Y^bar)^2/sum(Y - Y^bar)^2

- Adjusted R-squared = 1 - (1 - R^2) * (n-1/n-p-1)

	Where n = Number of observatons; p = Number of estimated parameters

- Validation metrics: MSE, RMSE, MAE, MAPE, SMAPE

---

**Logistic Regression:** When dependent variable is categorical we cannot use linear regression as linear regression assume dependent variable to follow normal distribution. In such case we may use logisitc regression instead of linear regression.

- Representation (incase of binary dependent variable) -> log(P(Y = 1)/P(Y = 0)) = X * beta + epsilon

	Where, log(P(Y = 1)/P(Y = 0)) is the log-odds of probability of success

- Transformation -> P(Y = 1) = 1/(1 + exp(-X * beta^hat))

- Cost function -> -(1/n) * sum(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))

- Gradient of the cost function -> X^t * (P(Y = 1) - Y)


* We need to use numerical optimization techniques like gradient descent or it's variants to minimize the above cost function w.r.t betas

* Logistic regression decision boundary is linear so it's a generalized linear regression. Eventually the decision of a sample depends on (X * beta^hat) value, where X * beta^hat is linear interms of beta.

- Validation metrics: Accuracy, Precision, Recall, F1 score, AUC-ROC

---

**Decision Trees:** It divides the data into subgroups iteratively until it reaches to the terminal node. Both regression and classification problems can be solved using decision trees.

- Representation -> Difficult to represent a decision tree because it is a non-linear model and also it trains rules to divide the data

- Cost function for regression (SSE) -> Sum of the squared errors across all training samples that fall within the subgroup

- Cost function for classification (Gini) -> Incase of binary classification it is the sum of `p * (1 - p)` of both left and right subgroup

**Intuition behind Gini:** `Y = 0/1` follows bernoulli radom variable with probability of success `p`. Where, p = proporton of records with Y = 1. `V(Y) = p * (1-p)`, If all of the records belongs to one class then `V(Y) = 0` and it is maximum when p = 0.5. So the purity of the node can be computed as sum of variance of left and right node dependent variables.

- Root node -> Decision tree is a greedy approach, in which we parse through each feature and it's candidate split points and select the one with lowest cost to start the decistion tree

- Children node -> Repeat the above process until we reach terminal node

- Terminal node -> We can check whether a node is terminal or not based on max depth of the tree and minimum samples to split hyper parameters

**Caution:** If we don't prune the decision tree then it's variance increases a lot and it can't be generalized to out-of-sample. If we prune decision tree too early then it can't train rules properly. Make sure you choose right hyper-parameter values to build a less variance and less bias decision tree.

---

**Random Forest:**

**Bagging:** Instead of building one good decision tree, bagging suggests to build n number of decision trees with different samples of the training set and combine the predicition with majority voting (classification) or mean (regression). Intuition behind considering different samples is, different views of the same problem.

**Cons:** Major drawback with bagging is that, it considers all features to train all models. So it's most likely that each tree might be using same features all the time

Random forest overcomes this drawback of bagging by sampling the data row-wise and also column-wise for each decision tree. Meaning, we will only consider sample of features to build each decision tree. It also suggests to consider sqrt(k) features to build each decision tree, where k is the total number features.

---

**Gradient Boost Machines:**

**Boosting:** Bagging builds multiple parallel decision trees (weak learners) and combine the predictions of all trees. Whereas in boosting we build one weak learner and improve that over the iterations

**Intuition behind boosting:** Boosting is very similar to human learning. We learn everything step-by-step. For instance, initially I was very bad at badminton and I started learning one thing at a time. I focused on short serve first and once I learnt that then I focused on long shots and then I focused on drop shots and so on. Everytime when I learn new technique I don't change/forget what I learnt in the past.

**Adaboost:** It starts with a decision stump (decision tree with one split) and tries to improve over the iterations by assigning more weightage to the observation which mis-classified in the previous iteration

	1. Initialize the equal weights to all observations (say, Wi = 1/N for all it, where N is the number of records)

	2. Build a decision stump

		* Compute loss (Gini incase of classification) for all features at all candidate split points and choose the one with minimum loss, and build a decision stump

	3. Compute performance of the stump: alpha = (1/2) * log((1 - Error)/Error). Where, Error is the proportion of misclassifications

	4. Update the weights of each sample: Wi = Wi * exp(alpha) if ith observation is misclassified and Wi = Wi * exp(-alpha) if ith observation is correctly classified

	5. Normalize the weights by dividing each weight with the sum of all weights

	6. Bucket these updated sample weights with bucket length equals to the weight

	7. Randomly select a sample of size N with replacement, use the updated weights as chance of getting selected in the sample

		* Observation which misclassified in the last iteration have higher weights so they will have higher probability to get selected multiple times in the sample
		
	8. Build a decision stump with the new data

	9. Repeat above steps till it reaches termination criteria


**Stagewise Additive Modeling:** Boosting model can be represented as an additive model as follows

							F(x) = sum(beta_m * b(x, gamma_m)) over m = 1 to M

- Where, beta_m and gamma_m are the model parameters. b(x, gamma_m) is a tree, and gamma_m parameterizes the splits

- Boosting is very different from other models, in other models we estimate beta_m and gamma_m jointly for all m = 1 to M. Whereas, in boosting we estimate (beta_0, gamma_0) in iteration 0 and it reamains fixed over the iterations. This process is called as stagewise modeling.

- General Loss function -> (beta_m, gamma_m) = sum(L(yi, F_m-1(x) + beta * b(x_i, gamma))) over i = 1 to N. This can be a log-loss incase of classification and mean squared error incase of regression   

							F_m(x) = F_m-1(x) + epsilon * beta_m * b(x_i, gamma_m)
	
- Where, epsilon is the shrinkage factor; b(x_i, gamma_m) is a tree, and gamma_m parameterizes the splits
	
**Gradient Boosting:** Gradient boosting converts this problem into a numerical optimization problem with objective is to minimize the loss of the model by adding weak learners using gradient descent like procedures. It uses functional gradients or gradients with functions.

 ---
 
**Navie Baye's classifier:** NBC is a generative model, all of the above discussed models are descriminative models. Descriminative model uses features which descriminant across the classes to build decision boundaries. Whereas, generative model models the distribution of input features to classify records.

It uses Baye's theorem to compute the probabilities as follows:

						P(Y belongs to C / X) = P(X / Y belongs to C) * P(Y belongs to C) / P(X)

- Where, P(X) does not depends on C so we can ignore this

**Assumption:** NBC assumes that all input features are categorical and it also assumes all input features are independent among themselves



