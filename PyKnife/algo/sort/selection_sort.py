# _*_ coding: utf-8 _*_

"""
Implementation of selection sort algorithm.

References:
[1] https://sort.hust.cc/2.selectionsort
[2] https://runestone.academy/runestone/books/published/pythonds/SortSearch/TheSelectionSort.html
[3] https://time.geekbang.org/column/article/41802

Author: Genpeng Xu
"""

from typing import List


class SelectionSort:
    @staticmethod
    def sort(arr: List[int]) -> None:
        n = len(arr)
        for i in range(n-1):
            # print("%d:" % i, end='')
            min_index = i
            for j in range(i+1, n):
                # print(" %d" % j, end='')
                if arr[j] < arr[min_index]:
                    min_index = j
            # print()
            if min_index != i:
                arr[i], arr[min_index] = arr[min_index], arr[i]


if __name__ == '__main__':
    arr = [5, 4, 3, 2, 1]
    print(arr)
    SelectionSort.sort(arr)
    print(arr)
