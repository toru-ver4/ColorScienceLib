#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transfer Functions モジュール のテスト
"""

import numpy as np
import unittest
import transfer_functions as tf


class TestEotf(unittest.TestCase):

    def test_gamma24(self):
        np.testing.assert_almost_equal(
            tf.eotf(0.0, tf.GAMMA24), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(0.5, tf.GAMMA24), 0.5 ** 2.4, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(1.0, tf.GAMMA24), 1.0, decimal=7)

    def test_sRGB(self):
        np.testing.assert_almost_equal(
            tf.eotf(0.0, tf.SRGB), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(0.5, tf.SRGB), 0.214041140482, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(1.0, tf.SRGB), 1.0, decimal=7)

    def test_hlg(self):
        np.testing.assert_almost_equal(
            tf.eotf(0.0, tf.HLG), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(0.5, tf.HLG), 0.0506970284911, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(1.0, tf.HLG), 1.00000002924, decimal=7)

    def test_st2084(self):
        np.testing.assert_almost_equal(
            tf.eotf(0.0, tf.ST2084), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(0.5, tf.ST2084), 0.00922457089941, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(1.0, tf.ST2084), 1.0, decimal=7)

    def test_slog3(self):
        np.testing.assert_almost_equal(
            tf.eotf(0.0, tf.SLOG3), -0.000365001428996, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(0.5, tf.SLOG3), 0.0108082585426, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(1.0, tf.SLOG3), 1.0, decimal=7)

    def test_logc(self):
        np.testing.assert_almost_equal(
            tf.eotf(0.0, tf.LOGC), -0.000313917050413, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(0.5, tf.LOGC), 0.00932075783426, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(1.0, tf.LOGC), 1.0, decimal=7)

    def test_log3g10(self):
        np.testing.assert_almost_equal(
            tf.eotf(0.0, tf.LOG3G10), -5.42527812174e-05, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(0.5, tf.LOG3G10), 0.00580893191444, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(1.0, tf.LOG3G10), 1.0, decimal=7)

    def test_log3g12(self):
        np.testing.assert_almost_equal(
            tf.eotf(0.0, tf.LOG3G12), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(0.5, tf.LOG3G12), 0.00197258876438, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(1.0, tf.LOG3G12), 1.0, decimal=7)

    def test_vlog(self):
        np.testing.assert_almost_equal(
            tf.eotf(0.0, tf.VLOG), -0.000484347897509, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(0.5, tf.VLOG), 0.00831820563111, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf(1.0, tf.VLOG), 1.0, decimal=7)


if __name__ == '__main__':
    pass
