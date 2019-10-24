# _*_ coding: utf-8 _*_

"""
Implementation of Bubble sort algorithm.

References:
[1] https://sort.hust.cc/1.bubblesort
[2] https://runestone.academy/runestone/books/published/pythonds/SortSearch/TheBubbleSort.html
[3] https://time.geekbang.org/column/article/41802

Author: Genpeng Xu
"""

from typing import List


class BubbleSort:
    @staticmethod
    def bubble_sort(arr: List[int]) -> None:
        n = len(arr)
        for i in range(n - 1):
            is_exchange = False
            for j in range(n - 1 - i):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
                    is_exchange = True
            if not is_exchange:
                break


if __name__ == '__main__':
    arr = [10, 8, 12, 15, 6, 3]
    print(arr)
    BubbleSort.bubble_sort(arr)
    print(arr)
