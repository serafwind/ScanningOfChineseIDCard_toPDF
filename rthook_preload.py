# -*- coding: utf-8 -*-
"""
运行时预加载钩子（在 PyInstaller 自动钩子之前执行）。

PyInstaller 的 PyQt5 自动运行时钩子会在主脚本运行前导入 PyQt5.QtCore，
而实测 onnxruntime 的核心 pyd 在 PyQt5 之后加载会触发 0xC0000005 段错误
（rembg 内部同样加载 onnxruntime）。因此在这里抢先导入，恢复安全的加载顺序。
"""

import onnxruntime  # noqa: F401
import rembg  # noqa: F401
