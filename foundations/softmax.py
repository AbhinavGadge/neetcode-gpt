import numpy as np
from numpy.typing import NDArray
import math


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        den = 0
        z_max = max(z)
        for num in z:
            den += math.exp(num - z_max)
        return [round(math.exp(num - z_max)/den, 4) for num in z]
