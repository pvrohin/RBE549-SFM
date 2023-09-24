import numpy as np

#Write a function to estimate the fundamental matrix between two images
def EstimateFundamentalMatrix(pts1, pts2):
    #Input:
    #pts1 and pts2 - Two Nx2 matrices containing n rows of [x,y] coordinates of matched points from two images
    #Output:
    #F - A 3x3 fundamental matrix containing the coefficients f_ij such that pts2.T * F * pts1 = 0
    #Note: This can be done using the 8-point algorithm (covered in class) by solving Af = 0. The resulting f
    #is a vector of 9 entries which can be arranged into a 3x3 matrix
    #Your code here
    A = np.zeros((pts1.shape[0], 9))
    for i in range(pts1.shape[0]):
        A[i, :] = [pts1[i, 0] * pts2[i, 0], pts1[i, 0] * pts2[i, 1], pts1[i, 0], pts1[i, 1] * pts2[i, 0], pts1[i, 1] * pts2[i, 1], pts1[i, 1], pts2[i, 0], pts2[i, 1], 1]
    U, S, V = np.linalg.svd(A)
    F = V[-1, :].reshape(3, 3)
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    return F
