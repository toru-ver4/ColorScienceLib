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


class TestOetf(unittest.TestCase):

    def test_gamma24(self):
        np.testing.assert_almost_equal(
            tf.oetf(0.0, tf.GAMMA24), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(0.5 ** 2.4, tf.GAMMA24), 0.5, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(1.0, tf.GAMMA24), 1.0, decimal=7)

    def test_sRGB(self):
        np.testing.assert_almost_equal(
            tf.oetf(0.0, tf.SRGB), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(0.214041140482, tf.SRGB), 0.5, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(1.0, tf.SRGB), 1.0, decimal=7)

    def test_hlg(self):
        np.testing.assert_almost_equal(
            tf.oetf(0.0, tf.HLG), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(0.0506970284911, tf.HLG), 0.5, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(1.00000002924, tf.HLG), 1.0, decimal=7)

    def test_st2084(self):
        np.testing.assert_almost_equal(
            tf.oetf(0.0, tf.ST2084), 7.30955903e-07, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(0.00922457089941, tf.ST2084), 0.5, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(1.0, tf.ST2084), 1.0, decimal=7)

    def test_slog3(self):
        np.testing.assert_almost_equal(
            tf.oetf(-0.000365001428996, tf.SLOG3), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(0.0108082585426, tf.SLOG3), 0.5, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(1.0, tf.SLOG3), 1.0, decimal=7)

    def test_logc(self):
        np.testing.assert_almost_equal(
            tf.oetf(-0.000313917050413, tf.LOGC), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(0.00932075783426, tf.LOGC), 0.5, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(1.0, tf.LOGC), 1.0, decimal=7)

    def test_log3g10(self):
        np.testing.assert_almost_equal(
            tf.oetf(-5.42527812174e-05, tf.LOG3G10), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(0.00580893191444, tf.LOG3G10), 0.5, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(1.0, tf.LOG3G10), 1.0, decimal=7)

    def test_log3g12(self):
        np.testing.assert_almost_equal(
            tf.oetf(0.0, tf.LOG3G12), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(0.00197258876438, tf.LOG3G12), 0.5, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(1.0, tf.LOG3G12), 1.0, decimal=7)

    def test_vlog(self):
        np.testing.assert_almost_equal(
            tf.oetf(-0.000484347897509, tf.VLOG), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(0.00831820563111, tf.VLOG), 0.5, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf(1.0, tf.VLOG), 1.0, decimal=7)


class TestEotfToLuminance(unittest.TestCase):

    def test_gamma24(self):
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.0, tf.GAMMA24), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.5, tf.GAMMA24), 0.5 ** 2.4 * 100, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(1.0, tf.GAMMA24), 1.0 * 100, decimal=7)

    def test_sRGB(self):
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.0, tf.SRGB), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.5, tf.SRGB), 0.214041140482 * 100,
            decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(1.0, tf.SRGB), 1.0 * 100, decimal=7)

    def test_hlg(self):
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.0, tf.HLG), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.5, tf.HLG), 0.0506970284911 * 1000,
            decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(1.0, tf.HLG) / 1000, 1.00000002924, decimal=7)

    def test_st2084(self):
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.0, tf.ST2084), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.5, tf.ST2084), 0.00922457089941 * 10000,
            decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(1.0, tf.ST2084), 1.0 * 10000, decimal=7)

    def test_slog3(self):
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.0, tf.SLOG3), -1.55818843738, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.5, tf.SLOG3), 46.1403768627, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(1.0, tf.SLOG3), 4268.99270413, decimal=7)

    def test_logc(self):
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.0, tf.LOGC), -1.72904182553, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.5, tf.LOGC), 51.3383396023, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(1.0, tf.LOGC), 5507.95766988, decimal=7)

    def test_log3g10(self):
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.0, tf.LOG3G10), -1.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.5, tf.LOG3G10), 107.071596775, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(1.0, tf.LOG3G10), 18432.234764, decimal=7)

    def test_log3g12(self):
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.0, tf.LOG3G12), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.5, tf.LOG3G12), 145.438670567, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(1.0, tf.LOG3G12), 73729.8484067, decimal=7)

    def test_vlog(self):
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.0, tf.VLOG), -2.23214285714, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(0.5, tf.VLOG), 38.3348898163, decimal=7)
        np.testing.assert_almost_equal(
            tf.eotf_to_luminance(1.0, tf.VLOG), 4608.55279567, decimal=7)


class TestOetfFromLuminance(unittest.TestCase):

    def test_gamma24(self):
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(0.0, tf.GAMMA24), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(0.5 ** 2.4 * 100, tf.GAMMA24), 0.5,
            decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(1.0 * 100, tf.GAMMA24), 1.0, decimal=7)

    def test_sRGB(self):
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(0.0, tf.SRGB), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(0.214041140482 * 100, tf.SRGB), 0.5,
            decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(1.0 * 100, tf.SRGB), 1.0, decimal=7)

    def test_hlg(self):
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(0.0, tf.HLG), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(0.0506970284911 * 1000, tf.HLG), 0.5,
            decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(1.00000002924 * 1000, tf.HLG), 1.0,
            decimal=7)

    def test_st2084(self):
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(0.0, tf.ST2084), 7.30955903e-07, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(0.00922457089941 * 10000, tf.ST2084), 0.5,
            decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(1.0 * 10000, tf.ST2084), 1.0, decimal=7)

    def test_slog3(self):
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(-1.55818843738, tf.SLOG3), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(46.1403768627, tf.SLOG3), 0.5, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(4268.99270413, tf.SLOG3), 1.0, decimal=7)

    def test_logc(self):
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(-1.72904182553, tf.LOGC), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(51.3383396023, tf.LOGC), 0.5, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(5507.95766988, tf.LOGC), 1.0, decimal=7)

    def test_log3g10(self):
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(-1.0, tf.LOG3G10), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(107.071596775, tf.LOG3G10), 0.5, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(18432.234764, tf.LOG3G10), 1.0, decimal=7)

    def test_log3g12(self):
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(0.0, tf.LOG3G12), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(145.438670567, tf.LOG3G12), 0.5, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(73729.8484067, tf.LOG3G12), 1.0, decimal=7)

    def test_vlog(self):
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(-2.23214285714, tf.VLOG), 0.0, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(38.3348898163, tf.VLOG), 0.5, decimal=7)
        np.testing.assert_almost_equal(
            tf.oetf_from_luminance(4608.55279567, tf.VLOG), 1.0, decimal=7)


if __name__ == '__main__':
    pass
