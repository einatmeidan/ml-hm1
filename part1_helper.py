# part 1, Q2
# first calculation

import numpy as np

Xt = np.array([
    [1, 1, 1, 1, 1],
    [1180, 2570, 770, 1960, 1680]
])

X = np.array([
    [1, 1180],
    [1, 2570],
    [1, 770],
    [1, 1960],
    [1, 1680]
])

XtX = Xt @ X
print("XtX =")
print(XtX)

# second calculation

# determinant of a 2x2 matrix [[a,b],[c,d]] is: ad - bc
det = XtX[0, 0] * XtX[1, 1] - XtX[0, 1] * XtX[1, 0]

# inverse formula for 2x2:
# (X^T X)^-1 = (1 / det) * [[d, -b],
#                           [-c, a]]

# matrix after changing places and signs inside the formula:
changed_matrix = np.array([
    [15254200, -8160],
    [-8160, 5]
])

second_calc = changed_matrix @ Xt

print("\nDeterminant (ad - bc) =")
print(det)

print("\nMatrix after change inside the inverse formula =")
print(changed_matrix)

print("\nX^T =")
print(Xt)

print("\nSecond calculation =")
print(second_calc)

# third calculation
# multiply by 1/det to get X dagger
inverse_by_formula = (1 / det) * second_calc

print("\nX† =")
print(inverse_by_formula)


# Q3
# first calculation for question 3:
# w* = X† y

y = np.array([
    [221900],
    [538000],
    [180000],
    [604000],
    [510000]
])

print("\ny =")
print(y)

w_star = inverse_by_formula @ y

print("\nw* = X†y =")
print(w_star)


# Q4
# compute the minimum squared loss: J(w*) = ||Xw* - y||^2

# first calculation for question 4:
# compute Xw*
y_pred = X @ w_star

print("\nXw* =")
print(y_pred)

# second calculation for question 4:
# compute the error vector Xw* - y
errors = y_pred - y

print("\nXw* - y =")
print(errors)

# third calculation for question 4:
# square each error
squared_errors = errors ** 2

print("\n(Xw* - y)^2 =")
print(squared_errors)

# final calculation:
# sum all squared errors
J = np.sum(squared_errors)

print("\nJ(w*) =")
print(J)