import numpy as np

def gaussian_elimination(A, b):
    A = A.astype(float) # 確保計算精度
    b = b.astype(float)
    n, m = A.shape

    if n >= m:
        raise ValueError("矩陣 A 的列數應該比行數多 (m > n)，否則系統不確定或無解。")

    # **組成增廣矩陣**
    Ab = np.hstack((A, b.reshape(-1, 1)))

    # **前向消去：將矩陣轉換為上三角形式**
    for i in range(n):
        # **選擇主元（Pivot）**
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]  # 交換行，確保數值穩定性

        # **歸一化主元**
        Ab[i] /= Ab[i, i]

        # **將下面的數字變為 0**
        for j in range(i + 1, n):
            Ab[j] -= Ab[j, i] * Ab[i]

    # **回代求解**
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, -1] - np.sum(Ab[i, i + 1:n] * x[i + 1:n])

    return x
