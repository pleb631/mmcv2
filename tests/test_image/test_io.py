import os
import os.path as osp
from pathlib import Path
from unittest.mock import patch

import cv2
import mmcv2
import numpy as np
import pytest
import torch
from mmcv2.fileio.file_client import HTTPBackend
from numpy.testing import assert_array_equal

if torch.__version__ == "parrots":
    pytest.skip("not necessary in parrots test", allow_module_level=True)


class TestIO:
    @classmethod
    def setup_class(cls):
        cls.data_dir = osp.join(osp.dirname(__file__), "../data")
        # the test img resolution is 400x300
        cls.img_path = osp.join(cls.data_dir, "color.jpg")
        cls.img_path_obj = Path(cls.img_path)
        cls.gray_img_path = osp.join(cls.data_dir, "grayscale.jpg")
        cls.gray_img_path_obj = Path(cls.gray_img_path)
        cls.gray_img_dim3_path = osp.join(cls.data_dir, "grayscale_dim3.jpg")
        cls.gray_alpha_img_path = osp.join(cls.data_dir, "gray_alpha.png")
        cls.palette_img_path = osp.join(cls.data_dir, "palette.gif")
        cls.exif_img_path = osp.join(cls.data_dir, "color_exif.jpg")
        cls.img = cv2.imread(cls.img_path)
        # http path
        cls.http_path = "http://path/of/your/file.jpg"

    def assert_img_equal(self, img, ref_img, ratio_thr=0.999):
        assert img.shape == ref_img.shape
        assert img.dtype == ref_img.dtype
        area = ref_img.shape[0] * ref_img.shape[1]
        diff = np.abs(img.astype("int32") - ref_img.astype("int32"))
        assert np.sum(diff <= 1) / float(area) > ratio_thr

    def test_imread(self):
        # HardDiskBackend
        img_cv2_color_bgr = mmcv2.imread(self.img_path)
        assert img_cv2_color_bgr.shape == (300, 400, 3)
        img_cv2_color_rgb = mmcv2.imread(self.img_path, channel_order="rgb")
        assert img_cv2_color_rgb.shape == (300, 400, 3)
        assert_array_equal(img_cv2_color_rgb[:, :, ::-1], img_cv2_color_bgr)
        img_cv2_grayscale1 = mmcv2.imread(self.img_path, "grayscale")
        assert img_cv2_grayscale1.shape == (300, 400)
        img_cv2_grayscale2 = mmcv2.imread(self.gray_img_path)
        assert img_cv2_grayscale2.shape == (300, 400, 3)
        img_cv2_unchanged = mmcv2.imread(self.gray_img_path, "unchanged")
        assert img_cv2_unchanged.shape == (300, 400)
        img_cv2_unchanged = mmcv2.imread(img_cv2_unchanged)
        assert_array_equal(img_cv2_unchanged, mmcv2.imread(img_cv2_unchanged))

        img_cv2_color_bgr = mmcv2.imread(self.img_path_obj)
        assert img_cv2_color_bgr.shape == (300, 400, 3)
        img_cv2_color_rgb = mmcv2.imread(self.img_path_obj, channel_order="rgb")
        assert img_cv2_color_rgb.shape == (300, 400, 3)
        assert_array_equal(img_cv2_color_rgb[:, :, ::-1], img_cv2_color_bgr)
        img_cv2_grayscale1 = mmcv2.imread(self.img_path_obj, "grayscale")
        assert img_cv2_grayscale1.shape == (300, 400)
        img_cv2_grayscale2 = mmcv2.imread(self.gray_img_path_obj)
        assert img_cv2_grayscale2.shape == (300, 400, 3)
        img_cv2_unchanged = mmcv2.imread(self.gray_img_path_obj, "unchanged")
        assert img_cv2_unchanged.shape == (300, 400)
        # HTTPBackend
        img_cv2_color_bgr = mmcv2.imread(self.img_path)
        with patch.object(HTTPBackend, "get", return_value=img_cv2_color_bgr) as mock_method:
            img_cv2_color_bgr_http = mmcv2.imread(self.http_path)
            img_cv2_color_bgr_http_with_args = mmcv2.imread(self.http_path, backend_args={"backend": "http"})
            mock_method.assert_called()
            assert_array_equal(img_cv2_color_bgr_http, img_cv2_color_bgr_http_with_args)

        with pytest.raises(FileNotFoundError):
            mmcv2.imread(osp.join(self.data_dir, "missing.jpg"))

        img_cv2_exif = mmcv2.imread(self.exif_img_path)
        assert img_cv2_exif.shape == (400, 300, 3)
        img_cv2_exif_unchanged = mmcv2.imread(self.exif_img_path, flag="unchanged")
        assert img_cv2_exif_unchanged.shape == (300, 400, 3)
        img_cv2_color_ignore_exif = mmcv2.imread(self.exif_img_path, flag="color_ignore_orientation")
        assert img_cv2_color_ignore_exif.shape == (300, 400, 3)
        img_cv2_grayscale_ignore_exif = mmcv2.imread(self.exif_img_path, flag="grayscale_ignore_orientation")
        assert img_cv2_grayscale_ignore_exif.shape == (300, 400)

    def test_imfrombytes(self):
        # OpenCV decoding, channel order: bgr
        with open(self.img_path, "rb") as f:
            img_bytes = f.read()
        img_cv2 = mmcv2.imfrombytes(img_bytes)
        assert img_cv2.shape == (300, 400, 3)

        # OpenCV decoding, channel order: rgb
        with open(self.img_path, "rb") as f:
            img_bytes = f.read()
        img_rgb_cv2 = mmcv2.imfrombytes(img_bytes, channel_order="rgb")
        assert img_rgb_cv2.shape == (300, 400, 3)
        assert_array_equal(img_rgb_cv2, img_cv2[:, :, ::-1])

        # backend cv2, grayscale, decode as 3 channels
        with open(self.gray_img_path, "rb") as f:
            img_bytes = f.read()
        gray_img_rgb_cv2 = mmcv2.imfrombytes(img_bytes)
        assert gray_img_rgb_cv2.shape == (300, 400, 3)

        # backend cv2, grayscale
        with open(self.gray_img_path, "rb") as f:
            img_bytes = f.read()
        gray_img_cv2 = mmcv2.imfrombytes(img_bytes, flag="grayscale")
        assert gray_img_cv2.shape == (300, 400)

        # backend cv2, grayscale dim3
        with open(self.gray_img_dim3_path, "rb") as f:
            img_bytes = f.read()
        gray_img_dim3_cv2 = mmcv2.imfrombytes(img_bytes, flag="grayscale")
        assert gray_img_dim3_cv2.shape == (300, 400)

        with pytest.raises(ValueError):
            mmcv2.imfrombytes(img_bytes, channel_order="unsupported_order")

    def test_imwrite(self, tmp_path):
        img = mmcv2.imread(self.img_path)
        out_file = str(tmp_path / "mmcv2_test.jpg")
        mmcv2.imwrite(img, out_file)
        rewrite_img = mmcv2.imread(out_file)
        os.remove(out_file)
        self.assert_img_equal(img, rewrite_img)

        with pytest.raises(cv2.error):
            mmcv2.imwrite(img, "error_file.jppg")
