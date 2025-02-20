import numpy as np

import gaussian_elimination

if __name__ == "__main__":
    A = np.array([
        [2, 0, 1, 1],
        [1, -2, 4, 7],
        [4, 1, -1, 0]
    ])

    b = np.array([1, 7, 0])

    example_a = np.array([
        [3, -1, 2, 5],
        [1, -2, -1, -10],
        [3, 0, -1, 0]
    ])

    example_b = np.array([5, -10, 0])

    # 用高斯消去法解 Ax = b
    exercise_solution = gaussian_elimination.gaussian_elimination(A, b)
    solution = gaussian_elimination.gaussian_elimination(example_a, example_b)

    print(f"解: {exercise_solution}")
    print(f"解: {solution}")
