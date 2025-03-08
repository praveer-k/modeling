import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from modeling.recommendation.helper import load_data


def get_loss(d: dict, W: np.ndarray, U: np.ndarray, b: np.ndarray, c: np.ndarray, mu: np.ndarray):
  # d: (user_id, movie_id) -> rating
  N = float(len(d))
  sse = 0
  for k, r in d.items():
    i, j = k
    p = W[i].dot(U[j]) + b[i] + c[j] + mu
    sse += (p - r)*(p - r)
  return sse / N

def update_W_and_b(W: np.ndarray, U: np.ndarray, b: np.ndarray, c: np.ndarray, mu: np.ndarray, N: int, K: int, reg: int, user2movie: dict, usermovie2rating: dict):
  # update W and b
  t0 = datetime.now()
  for i in range(N):
    # for W
    matrix = np.eye(K) * reg
    vector = np.zeros(K)
    # for b
    bi = 0
    for j in user2movie[i]:
      r = usermovie2rating[(i,j)]
      matrix += np.outer(U[j], U[j])
      vector += (r - b[i] - c[j] - mu)*U[j]
      bi += (r - W[i].dot(U[j]) - c[j] - mu)
    # set the updates
    W[i] = np.linalg.solve(matrix, vector)
    b[i] = bi / (len(user2movie[i]) + reg)
    if i % (N//10) == 0:
      print("i:", i, "N:", N)
  print("updated W and b:", datetime.now() - t0)

def update_U_and_c(W: np.ndarray, U: np.ndarray, b: np.ndarray, c: np.ndarray, mu: np.ndarray, M: int, K: int, reg: int, movie2user: dict, usermovie2rating: dict):
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
        r = usermovie2rating[(i,j)]
        matrix += np.outer(W[i], W[i])
        vector += (r - b[i] - c[j] - mu)*W[i]
        cj += (r - W[i].dot(U[j]) - b[i] - mu)
      # set the updates
      U[j] = np.linalg.solve(matrix, vector)
      c[j] = cj / (len(movie2user[j]) + reg)
      if j % (M//10) == 0:
        print("j:", j, "M:", M)
    except KeyError:
      # possible not to have any ratings for a movie
      pass
  print("updated U and c:", datetime.now() - t0)

def get_dimensions(user2movie: dict, movie2user: dict, usermovie2rating_test: dict) -> tuple[int, int]:
  N = np.max(list(user2movie.keys())) + 1
  # the test set may contain movies the train set doesn't have data on
  m1 = np.max(list(movie2user.keys()))
  m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
  M = max(m1, m2) + 1
  print("N:", N, "M:", M)
  return N, M

def train_for_recommendation(data_dir: Path) -> tuple[list, list]:
  # initialize variables
  user2movie, movie2user, usermovie2rating, usermovie2rating_test = load_data(data_dir)
  N, M = get_dimensions(user2movie, movie2user, usermovie2rating_test)
  K = 10 # latent dimensionality
  W = np.random.randn(N, K)
  b = np.zeros(N)
  U = np.random.randn(M, K)
  c = np.zeros(M)
  mu = np.mean(list(usermovie2rating.values()))
  # train the parameters
  epochs = 25
  reg = 20. # regularization penalty
  train_losses = []
  test_losses = []
  for epoch in range(epochs):
    print("epoch:", epoch)
    epoch_start = datetime.now()
    # perform updates
    # prediction[i,j] = W[i].dot(U[j]) + b[i] + c.T[j] + mu
    update_W_and_b(W, U, b, c, mu, N, K, reg, user2movie, usermovie2rating)
    update_U_and_c(W, U, b, c, mu, M, K, reg, movie2user, usermovie2rating)
    print("epoch duration:", datetime.now() - epoch_start)
    # store train loss
    t0 = datetime.now()
    train_losses.append(get_loss(usermovie2rating))
    # store test loss
    test_losses.append(get_loss(usermovie2rating_test))
    print("calculate cost:", datetime.now() - t0)
    print("train loss:", train_losses[-1])
    print("test loss:", test_losses[-1])
  print("train losses:", train_losses)
  print("test losses:", test_losses)
  return train_losses, test_losses

def plot_train_test_loss(train_losses: list, test_losses: list):
  # plot losses
  plt.plot(train_losses, label="train loss")
  plt.plot(test_losses, label="test loss")
  plt.legend()
  plt.show()