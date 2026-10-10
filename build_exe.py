# -*- coding: utf-8 -*-
"""
打包脚本：把程序打包为单文件exe

规避一个已知问题：PyInstaller 在 Windows 上做二进制依赖扫描时，
会强制把 Qt 绑定（PyQt5）最先导入。实测 onnxruntime 的核心 pyd 在
PyQt5 之后加载会触发 0xC0000005 访问冲突（rembg 同理，它内部也加载
onnxruntime）；先加载 onnxruntime 则一切正常（main.py 顶部的导入顺序
即按此处理）。

因此在扫描子进程里抢先导入 onnxruntime 和 rembg，再让 Qt 正常导入。
"""
import PyInstaller.isolated as isolated

_orig_call = isolated.Python.call
_preload_done = False


def _patched_call(self, func, *args, **kwargs):
    global _preload_done
    if getattr(func, '__name__', '') == 'import_library' and not _preload_done:
        _preload_done = True
        # onnxruntime/rembg 必须先于 Qt 绑定导入，避免 DLL 加载顺序段错误
        _orig_call(self, func, 'onnxruntime')
        _orig_call(self, func, 'rembg')
    return _orig_call(self, func, *args, **kwargs)


isolated.Python.call = _patched_call

from PyInstaller.__main__ import run

run([
    '--noconfirm',
    '--onefile',
    '--windowed',
    '--name', '身份证扫描件生成器',
    '--add-data', 'models;models',
    '--runtime-hook', 'rthook_preload.py',
    '--collect-all', 'onnxruntime',
    '--collect-all', 'rembg',
    # 导入链中会查询版本元数据的包（pymatting 等缺失元数据会启动失败）
    '--copy-metadata', 'rembg',
    '--copy-metadata', 'numpy',
    '--copy-metadata', 'pymatting',
    '--copy-metadata', 'tqdm',
    '--copy-metadata', 'paramiko',
    '--copy-metadata', 'invoke',
    'main.py',
])
