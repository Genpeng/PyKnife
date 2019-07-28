# _*_ coding: utf-8 _*_

"""
Implement Fibonacci function by using three methods.

Author: Genpeng Xu
"""

import time
from typing import List


class Solution1:
    def fib(self, n: int) -> int:
        """Recurrently calculate Fibonacci sequence.

        Arguments:
            n: int, the input integer

        Returns:
            int, the result of `Fib(n)`
        """
        if n == 0:
            return 0
        if n == 1:
            return 1
        return self.fib(n - 1) + self.fib(n - 2)


class Solution2:
    def fib(self, n: int) -> int:
        """Calculate Fibonacci sequence by using *Memoization*.

        Arguments:
            n: int, the input integer

        Returns:
            int, the result of `Fib(n)`
        """
        memo = [-1] * (n + 1)
        return self._fib(n, memo)

    def _fib(self, n: int, memo: List[int]) -> int:
        if n == 0:
            return 0
        if n == 1:
            return 1
        if memo[n] == -1:
            memo[n] = self._fib(n - 1, memo) + self._fib(n - 2, memo)
        return memo[n]


class Solution3:
    def fib(self, n: int) -> int:
        """Calculate Fibonacci sequence by using *Dynamic Programming*.

        Arguments:
            n: int, the input integer

        Returns:
            int, the result of `Fib(n)`
        """
        memo = [0] * (n + 1)
        memo[1] = 1
        for i in range(2, n + 1):
            memo[i] = memo[i - 1] + memo[i - 2]
        return memo[n]


if __name__ == '__main__':
    n = 6
    solution = Solution2()
    start_time = time.time()
    print("[INFO] Fib(%d) = %d" % (n, solution.fib(n)))
    print("[INFO] Done in %f seconds." % (time.time() - start_time))
