# Data Science Preparation

## Optimization

**Objective:** Minimization of cost function or loss function

**Optimization process:** Update parameters iteratively to reach local or global minima of cost function

**Batch Gradient Descent or Gradient Descent:** Update parameters in the opposite direction of the gradient of the cost function by considering whole data at once

![Gradient Descent Updates](https://bit.ly/3hynJpo)

Theta = Theta - lr * gradient(cost function w.r.t theta given whole data)

	Number of records = 100; Number of epochs = 10 => Number of parameter updates = 10
	
	Hyper parameter: learning rate

	Cons:

		- If the training data is too big then it's difficult to fit the entire data into memory for gradient computation
		- Redundent gradient computation for similar examples
		- Online learning is not possible as parameters updates with whole data
