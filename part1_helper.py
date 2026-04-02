#part 1, Q2:
#first calculation 

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

Xt = np.array([
    [1, 1, 1, 1, 1],
    [1180, 2570, 770, 1960, 1680]
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

# now multiply by (1 / det)
inverse_by_formula = (1 / det) * second_calc

np.set_printoptions(suppress=True)

print("\nThird calculation =")
print(inverse_by_formula)