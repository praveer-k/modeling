import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from modeling.config import logger
from modeling.recommendation.helper import load_data


def get_loss(d: dict, W: np.ndarray, U: np.ndarray, b: np.ndarray, c: np.ndarray, mu: np.ndarray) -> float:
    """
    Args:
      - d (dict): user, movie pairs to rating
      - W (np.ndarray): users embedding matrix
      - U (np.ndarray): movies embedding matrix
      - b (np.ndarray): user bias
      - c (np.ndarray): movie bias
      - mu (np.ndarray): global bias

    Returns:
      _type_: loss scalar value

    The function calculates


    $$\text{Loss} = \frac{1}{N} \sum_{(i, j) \in d} \left( W_i \cdot U_j + b_i + c_j + \mu - r_{ij} \right)^2$$
    
    Where:
      - $$d$$ is a dictionary mapping (user, movie) pairs to ratings $$r_{ij}$$
      - $$W$$ and $$U$$ are embedding matrices for users and movies.
      - $$b$$ and $$c$$ are bias terms for users and movies.
      - $$\mu$$ is a global bias (scalar).
      - $$N$$ is the total number of ratings in the dataset.
    
    Python Implementation:
      - Iterate over each (user, movie) pair in d, extracting the rating r.
      - Predicts the rating using:

        $$p=W[i]⋅U[j]+b[i]+c[j]+\mu$$
      - Computes squared error and accumulates it.
      - Returns the mean squared error (MSE).

    Potential Improvements:
      - Could use NumPy vectorization to improve performance instead of looping.
      - Adding regularization would help avoid overfitting.
    """
    N = float(len(d))
    sse = 0
    for k, r in d.items():
        i, j = k
        p = W[i].dot(U[j]) + b[i] + c[j] + mu
        sse += (p - r) * (p - r)
    return sse / N


def update_W_and_b(
    W: np.ndarray,
    U: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    mu: np.ndarray,
    N: int,
    K: int,
    reg: int,
    user2movie: dict,
    usermovie2rating: dict,
):
    # update W and b
    t0 = datetime.now()
    for i in range(N):
        # for W
        matrix = np.eye(K) * reg
        vector = np.zeros(K)
        # for b
        bi = 0
        for j in user2movie[i]:
            r = usermovie2rating[(i, j)]
            matrix += np.outer(U[j], U[j])
            vector += (r - b[i] - c[j] - mu) * U[j]
            bi += r - W[i].dot(U[j]) - c[j] - mu
        # set the updates
        W[i] = np.linalg.solve(matrix, vector)
        b[i] = bi / (len(user2movie[i]) + reg)
        if i % (N // 10) == 0:
            logger.info(f"i:{i} N:{N}")
    logger.info(f"updated W and b: {datetime.now() - t0}")


def update_U_and_c(
    W: np.ndarray,
    U: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    mu: np.ndarray,
    M: int,
    K: int,
    reg: int,
    movie2user: dict,
    usermovie2rating: dict,
):
    # update U and c
    t0 = datetime.now()
    for j in range(M):
        # for U
        matrix = np.eye(K) * reg
        vector = np.zeros(K)
        # for c
        cj = 0
        try:
            for i in movie2user[j]:
                r = usermovie2rating[(i, j)]
                matrix += np.outer(W[i], W[i])
                vector += (r - b[i] - c[j] - mu) * W[i]
                cj += r - W[i].dot(U[j]) - b[i] - mu
            # set the updates
            U[j] = np.linalg.solve(matrix, vector)
            c[j] = cj / (len(movie2user[j]) + reg)
            if j % (M // 10) == 0:
                logger.info(f"j:{j} M:{M}")
        except KeyError:
            # possible not to have any ratings for a movie
            pass
    logger.info(f"updated U and c: {datetime.now() - t0}")


def get_dimensions(
    user2movie: dict, movie2user: dict, usermovie2rating_test: dict
) -> tuple[int, int]:
    N = np.max(list(user2movie.keys())) + 1
    # the test set may contain movies the train set doesn't have data on
    m1 = np.max(list(movie2user.keys()))
    m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
    M = max(m1, m2) + 1
    logger.info(f"N:{N} M:{M}")
    return N, M


def train_for_recommendation(data_dir: Path) -> tuple[list, list]:
    # initialize variables
    user2movie, movie2user, usermovie2rating, usermovie2rating_test = load_data(
        data_dir
    )
    N, M = get_dimensions(user2movie, movie2user, usermovie2rating_test)
    K = 10  # latent dimensionality
    W = np.random.randn(N, K)
    b = np.zeros(N)
    U = np.random.randn(M, K)
    c = np.zeros(M)
    mu = np.mean(list(usermovie2rating.values()))
    # train the parameters
    epochs = 25
    reg = 20.0  # regularization penalty
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        logger.info(f"epoch:{epoch}")
        epoch_start = datetime.now()
        # perform updates
        # prediction[i,j] = W[i].dot(U[j]) + b[i] + c.T[j] + mu
        update_W_and_b(W, U, b, c, mu, N, K, reg, user2movie, usermovie2rating)
        update_U_and_c(W, U, b, c, mu, M, K, reg, movie2user, usermovie2rating)
        logger.info(f"epoch duration:{datetime.now() - epoch_start}")
        # store train loss
        t0 = datetime.now()
        train_losses.append(get_loss(usermovie2rating, W, U, b, c, mu))
        # store test loss
        test_losses.append(get_loss(usermovie2rating_test, W, U, b, c, mu))
        logger.info(f"calculate cost:{datetime.now() - t0}")
        logger.info(f"train loss:{train_losses[-1]}")
        logger.info(f"test loss:{test_losses[-1]}")
    logger.info(f"train losses:{train_losses}")
    logger.info(f"test losses:{test_losses}")
    return train_losses, test_losses


def plot_train_test_loss(train_losses: list, test_losses: list):
    # plot losses
    plt.plot(train_losses, label="train loss")
    plt.plot(test_losses, label="test loss")
    plt.legend()
    plt.show()
