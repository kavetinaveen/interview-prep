# Data Science Preparation

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
