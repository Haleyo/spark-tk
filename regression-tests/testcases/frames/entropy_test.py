""" Test Shannon entropy calculations """
import unittest
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from qalib import sparktk_test


class EntropyTest(sparktk_test.SparkTKTestCase):

    def test_entropy_coin_flip(self):
        """ Get entropy on balanced coin flip. """
        frame_load = 10 * [['H'], ['T']]
        expected = math.log(2)

        frame = self.context.frame.create(frame_load, schema=[("data", str)])
        computed_entropy = frame.entropy("data")
        self.assertAlmostEqual(computed_entropy, expected, delta=.001)

    def test_entropy_exponential(self):
        """ Get entropy on exponential distribution. """
        frame_load = [[0, 1], [1, 2], [2, 4], [4, 8]]
        # Expected result is from an on-line entropy calculator in base 2.
        expected = 1.640223928941852 * math.log(2)

        frame = self.context.frame.create(frame_load, schema=[("data", int), ("weight", int)])

        computed_entropy = frame.entropy("data", "weight")
        self.assertAlmostEqual(computed_entropy, expected)


if __name__ == '__main__':
    unittest.main()