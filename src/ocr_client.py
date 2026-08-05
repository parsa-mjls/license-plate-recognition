# -*- coding: utf-8 -*-
"""
ocr_client.py
=============
Client-side wrapper that starts ocr_worker.py as a separate subprocess and
talks to it over stdin/stdout.

Why a subprocess at all?
-------------------------
PaddleOCR/Paddle and PyTorch (used by the YOLO detector) can conflict when
imported in the same Python process on some platforms/driver combinations.
Running PaddleOCR in its own process keeps the two frameworks fully isolated,
at the cost of a small IPC overhead per crop.
"""

import os
import sys
import subprocess
import tempfile

import cv2


class PaddleOCRClient:
    def __init__(self, worker_script_path, python_executable=None):
        self.python_executable = python_executable or sys.executable
        self.worker_script_path = str(worker_script_path)
        self.process = None
        self._tmp_dir = tempfile.mkdtemp(prefix="plate_ocr_")
        self._tmp_img_path = os.path.join(self._tmp_dir, "crop.png")

    def start(self):
        print("[OCR] Starting PaddleOCR worker process (this may take a few seconds)...")

        # Force UTF-8 for the worker's own stdin/stdout, regardless of the
        # OS default codepage (e.g. cp1252 on Windows). Without this, writing
        # Persian/Arabic plate text from ocr_worker.py can crash with a
        # UnicodeEncodeError as soon as the first real result is produced.
        worker_env = os.environ.copy()
        worker_env["PYTHONIOENCODING"] = "utf-8"

        self.process = subprocess.Popen(
            [self.python_executable, "-u", self.worker_script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # IMPORTANT: stderr is intentionally NOT merged into stdout.
            # stdout is a strict one-line-per-response protocol channel; if
            # PaddleOCR/PaddlePaddle print warnings, logs, or a raw traceback
            # to stderr, merging them into stdout would desync every
            # subsequent response (each stray line would get consumed as the
            # "answer" to the next plate). Leaving stderr=None lets those
            # messages print directly to this console instead, purely for
            # visibility/debugging.
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=worker_env,
        )

        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(
                    "OCR worker process exited before signaling readiness. "
                    "Check its output above for the actual error."
                )
            line = line.strip()
            if line == "READY":
                break
            print("[ocr_worker]", line)

        print("[OCR] Worker ready.")

    def ocr(self, image_bgr):
        """Send a cropped plate image (numpy array) and get back (text, confidence)."""
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("OCR worker process is not available.")

        cv2.imwrite(self._tmp_img_path, image_bgr)

        self.process.stdin.write(self._tmp_img_path + "\n")
        self.process.stdin.flush()

        result_line = self.process.stdout.readline()
        if not result_line:
            raise RuntimeError("OCR worker process closed without responding.")

        result_line = result_line.strip()
        if "\t" in result_line:
            text, conf_str = result_line.split("\t", 1)
            try:
                conf = float(conf_str)
            except ValueError:
                conf = 0.0
        else:
            text, conf = result_line, 0.0

        return text, conf

    def close(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
        try:
            if os.path.exists(self._tmp_img_path):
                os.remove(self._tmp_img_path)
            os.rmdir(self._tmp_dir)
        except OSError:
            pass
