# _*_ coding: utf-8 _*_


def merge_sort(arr: list):
    """
    Sorting the list by using merge sort.

    Parameters
    ----------
    arr: list, the list needs to be sorted
    """
    _merge_sort(arr, 0, len(arr) - 1)


def _merge_sort(arr: list, li: int, ri: int):
    """
    Sorting the list by using merge sort, where the start index is `li` and the end index is `ri`.

    Parameters
    ----------
    arr: list, the list needs to be sorted
    li: int, the start index of the list
    ri: int, the end index of the list
    """
    if li == ri:
        return
    mi = li + (ri - li) // 2
    _merge_sort(arr, li, mi)
    _merge_sort(arr, mi + 1, ri)
    merge(arr, li, mi + 1, ri)


def merge(arr: list, li: int, mi: int, ri: int):
    """
    Merge two smaller sorted list into a large list, where the index of one list starts from
    `li` to `mi - 1`, and the index of the other list starts from `mi` to `ri`.

    Parameters
    ----------
    arr: list, the big list contains two smaller sorted list
    li: int, the start index of the left half list
    mi: int, the start index of the right half list
    ri: int, the end index of the right half list
    """
    right_half = arr[mi:]
    i, j, k = mi - 1, ri - mi, ri
    while i >= li and j >= 0:
        if arr[i] > right_half[j]:
            arr[k] = arr[i]
            i -= 1
        else:
            arr[k] = right_half[j]
            j -= 1
        k -= 1
    while j >= 0:
        arr[k] = right_half[j]
        j -= 1
        k -= 1


if __name__ == '__main__':
    arr = [55, 75, 89, 36, 22, 18, 90, 66, 78]
    print(arr)
    merge_sort(arr)
    print(arr)

