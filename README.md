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
