# _*_ coding: utf-8 _*_

"""
Implementation of quick sort algorithm.

Reference:
[1] https://runestone.academy/runestone/books/published/pythonds/SortSearch/TheQuickSort.html

Author: Genpeng Xu
"""

from typing import List


def _quick_sort(nums: List[int], li: int, ri: int) -> None:
    if li >= ri:
        return
    split_point = _partition(nums, li, ri)
    _quick_sort(nums, li, split_point - 1)
    _quick_sort(nums, split_point + 1, ri)


def _partition(nums: List[int], li: int, ri: int) -> int:
    pivot_value = nums[li]
    left_mark, right_mark = li + 1, ri
    done = False
    while not done:
        while left_mark <= right_mark and nums[left_mark] <= pivot_value:
            left_mark += 1
        while left_mark <= right_mark and nums[right_mark] >= pivot_value:
            right_mark -= 1
        if right_mark < left_mark:
            done = True
        else:
            tmp = nums[left_mark]
            nums[left_mark] = nums[right_mark]
            nums[right_mark] = tmp
    tmp = nums[li]
    nums[li] = nums[right_mark]
    nums[right_mark] = tmp
    return right_mark


class QuickSort:
    @staticmethod
    def quick_sort(nums: List[int]) -> None:
        _quick_sort(nums, 0, len(nums) - 1)


if __name__ == "__main__":
    from time import time
    n = 300
    nums = [i for i in range(1, n + 1)]
    t0 = time()
    QuickSort.quick_sort(nums)
    print("[INFO] Done in %.4f seconds" % (time() - t0))
