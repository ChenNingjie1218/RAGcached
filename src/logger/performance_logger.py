# performance_logger.py
import time
import json
import threading
from typing import Dict, Any
from src.configs.config import performance_log_path
import os
import atexit

class PerformanceLogger:
    _instance_lock = threading.Lock()
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, log_file="performance_log.json", buffer_size=100):
        # 防止重复初始化
        if hasattr(self, '_initialized'):
            return
        self.log_file = os.path.join(performance_log_path, log_file)
        self.logs = []
        self.buffer_size = buffer_size
        self.lock = threading.Lock()  # 线程安全
        self._flushing_thread = None
        self._initialized = True

        # 注册退出钩子，保证最后一批日志写入
        atexit.register(self.flush_sync)

    def record(self, module: str, event: str, data: Dict[str, Any]):
        # 获取格式化时间：年-月-日 时:分:秒
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        entry = {
            "time": current_time,
            "module": module,
            "event": event,
            "data": data
        }

        with self.lock:
            self.logs.append(entry)

        # 达到 buffer 大小就触发 flush
        if len(self.logs) >= self.buffer_size:
            self.async_flush()

    def async_flush(self):
        """启动一个线程执行刷盘操作"""
        if not self.logs:
            return

        # 防止重复创建线程
        if self._flushing_thread and self._flushing_thread.is_alive():
            return

        self._flushing_thread = threading.Thread(target=self._do_flush, daemon=True)
        self._flushing_thread.start()

    def _do_flush(self):
        """实际写入磁盘的方法"""
        with self.lock:
            logs_to_write = self.logs[:]
            self.logs.clear()

        if not logs_to_write:
            return

        try:
            with open(self.log_file, "a") as f:
                for entry in logs_to_write:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[ERROR] 写入日志失败: {e}")

    def flush_sync(self):
        """同步强制刷盘（如程序退出时调用）"""
        self._do_flush()

    @classmethod
    def reset_logger(cls, log_file: str = None, buffer_size: int = None):
        """
        重置当前单例实例
        :param log_file: 新的日志文件路径
        :param buffer_size: 新的缓冲大小
        """
        if cls._instance is not None:
            cls._instance.flush_sync()  # 刷盘旧日志

        params = {}
        if log_file:
            params["log_file"] = log_file
        if buffer_size is not None:
            params["buffer_size"] = buffer_size

        cls._instance = cls(**params)

    @classmethod
    def record_event(cls, module: str, event: str, data: dict):
        """对外暴露的类方法"""
        instance = cls()
        instance.record(module, event, data)
