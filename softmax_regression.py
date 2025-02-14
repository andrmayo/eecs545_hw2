"""EECS545 HW2: Softmax Regression."""

import numpy as np
import math


def hello():
    print('Hello from softmax_regression.py')


def compute_softmax_probs(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Computes probabilities for logit x being each class.

    Inputs:
      - X: Numpy array of shape (num_data, num_features).
      - W: Numpy array of shape (num_class, num_features). The last row is a zero vector.
    Returns:
      - probs: Numpy array of shape (num_data, num_class). The softmax
        probability with respect to W.
    """
    probs = None
    ###########################################################################
    # TODO: compute softmax probability of X with respect to W and store the  #
    # Softmax probability out to 'probs'.                                     #
    # If you are not careful here, it is easy to run into numeric instability #
    # (Check Numeric Stability in http://cs231n.github.io/linear-classify/)   #
    # Hint: the pseudo code in the link will not be 100% matched with ours.   #
    # You may need to slightly edit the code script in the link.              #
    ###########################################################################
    # shift matrix values so that highest value is 0 to avoid numerical instability
    shifted = X @ W.T
    # let's try doing the shifting with a loop
    for i, row in enumerate(shifted):
        row_max = np.max(row)
        shifted[i] -= row_max
        
    shifted = np.exp(shifted)

    #shifted -= np.tile(np.reshape(np.max(shifted, axis = 1), shape=(-1, 1)), (1, shifted.shape[1]))
    #shifted = np.exp(shifted)

    probs = np.einsum("ij, i -> ij", shifted, 1/(np.sum(shifted, axis = 1))) 


    #probs = np.einsum("ij, i -> ij", np.exp(X @ W.T), 1/np.sum(np.exp(X @ W.T), axis = 1)) 
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return probs


def gradient_ascent_train(X_train: np.ndarray,
                          Y_train: np.ndarray,
                          num_class: int,
                          max_iters: int = 300) -> np.ndarray:
    """Computes w from the train set (X_train, Y_train).
    This implementation uses gradient ascent algorithm derived from the previous question.

    Inputs:
      - X_train: Numpy array of shape (num_data, num_features).
                 Please consider this input as \\phi(x) (feature vector).
      - Y_train: Numpy array of shape (num_data, 1) that has class labels in
                 [1 .. num_class].
      - num_class: Number of class labels
      - max_iters: Maximum number of iterations
    Returns:
      - W: Numpy array of shape (num_class, num_features). The last row is a zero vector.
           We will use the trained weights on the test set to measure the performance.
    """
    N, d = X_train.shape  # the number of samples in training dataset, dimension of feature
    W = np.zeros((num_class, d), dtype=X_train.dtype)
    class_matrix = np.eye(num_class, dtype=W.dtype)

    int_Y_train = Y_train.astype(np.int32)
    alpha = 0.0005
    count_c = 0
    for epoch in range(max_iters):
        # A single iteration over all training examples
        delta_W = np.zeros((num_class, d), dtype=W.dtype)
        ###################################################################
        # TODO: Compute the cumulated weight 'delta_W' for each point.    #
        # You are allowed to use compute_softmax_probs function.          #
        # Note that Y_train has class labels in [1 ~ num_class]           #
        ###################################################################
        class_probs = -compute_softmax_probs(X_train, W)
        for i, label in enumerate(Y_train):
            class_probs[i, int(label[0] - 1)] += 1
        class_probs = X_train.T @ class_probs
        delta_W = class_probs.T
        #for i in range(num_class-1): 
            #class_indices = np.where(np.reshape(Y_train, shape=-1) == i+1)
            #inverse_indices = np.where(np.reshape(Y_train, shape=-1) != i+1)
            #X_inclass = X_train.copy()
            #X_inclass[inverse_indices] = 0

            #m_probabilities = np.reshape(class_probs[:, i], shape=-1)
            #delta_W[i] += np.sum(X_inclass - X_train * np.column_stack([m_probabilities]*X_train.shape[1]), axis = 0)
            #delta_W[i] += (1 - (np.exp(X_train[class_indices, :] @ W[i]) / np.sum(np.exp(X @ W.T), axis = 1))) @ X_train[class_indices, :]
            #delta_W[i] -= (np.exp(X_train[inverse_indices, :] @ W[i]) / np.sum(np.exp(X @ W.T), axis = 1)) @ X_train[inverse_indices, :] 

        ###################################################################
        #                        END OF YOUR CODE                         #
        ###################################################################
        W_new = W + alpha * delta_W
        W[:num_class-1, :] = W_new[:num_class-1, :]

        # Stopping criteria
        count_c += 1 if epoch > 300 and np.sum(abs(alpha * delta_W)) < 0.05 else 0
        if count_c > 5:
            break

    return W


def compute_accuracy(X_test: np.ndarray,
                     Y_test: np.ndarray,
                     W: np.ndarray,
                     num_class: int) -> float:
    """Computes the accuracy of trained weight W on the test set.

    Inputs:
      - X_test: Numpy array of shape (num_data, num_features).
      - Y_test: Numpy array of shape (num_data, 1) consisting of class labels
                in the range [1 .. num_class].
      - W: Numpy array of shape (num_class, num_features).
      - num_class: Number of class labels
    Returns:
      - accuracy: accuracy value in 0 ~ 1.
    """
    count_correct = 0
    N_test = Y_test.shape[0]
    int_Y_test = Y_test.astype(np.int32)
    ###########################################################################
    # TODO: save the number of correct prediction to 'count_correct' variable.#
    # We are using this value at the end of this function by dividing it to   #
    # number of (X, Y) data pairs. Hint: check the equation in the homework.  #
    ###########################################################################
    probs = compute_softmax_probs(X_test, W)
    predictions = (np.argmax(probs, axis = 1) + 1).astype(np.int32) 
    int_Y_test = np.reshape(int_Y_test, -1)
    count_correct = np.sum(predictions == int_Y_test, dtype = np.float32)
    #count_correct = np.array(predictions[predictions==int_Y_test].size)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

    accuracy = float(count_correct / (N_test * 1.0))
    return accuracy
