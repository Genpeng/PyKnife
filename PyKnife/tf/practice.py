# _*_ coding: utf-8 _*_

"""

Author: Genpeng Xu
Date:   2019/03/24
"""

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    a = tf.constant(2, dtype=tf.float32, name='a')
    b = tf.constant(3, dtype=tf.float32, name='b')

    x = tf.add(a, b, name='add')

    writer = tf.summary.FileWriter("./graph/test/", tf.get_default_graph())
    with tf.Session() as sess:
        print(sess.run(x))

    writer.close()


if __name__ == '__main__':
    main()
