# _*_ coding: utf-8 _*_

"""
This is the solution of no.703 problem in the LeetCode,
where the website of the problem is as follow:
https://leetcode.com/problems/kth-largest-element-in-a-stream/

Author: StrongXGP
Date:   2018/12/06
"""

import heapq


class KthLargest:

    def __init__(self, k, nums):
        """
        :type k: int
        :type nums: List[int]
        """
        self._pq = []
        self._k = k

        if not nums:
            return

        for num in nums:
            if len(self._pq) < k:
                heapq.heappush(self._pq, num)
            elif num > self._pq[0]:
                heapq.heapreplace(self._pq, num)

    def add(self, val):
        """
        :type val: int
        :rtype: int
        """
        if len(self._pq) < self._k - 1:
            raise Exception("[ERROR] The number of elements is less than (k - 1)!")
        elif len(self._pq) == self._k - 1:
            heapq.heappush(self._pq, val)
        elif val > self._pq[0]:
            heapq.heapreplace(self._pq, val)
        return self._pq[0]

# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)


def main():
    k = 3
    arr = [4, 5, 8, 2]
    kth_largest = KthLargest(k, arr)
    print(kth_largest.add(3))
    print(kth_largest.add(5))
    print(kth_largest.add(10))
    print(kth_largest.add(9))
    print(kth_largest.add(4))


if __name__ == '__main__':
    main()
