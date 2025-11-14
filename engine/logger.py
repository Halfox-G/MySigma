import os
import sys
import logging

# 移除对 utils 的依赖
# from utils import pyt_utils 

_default_level_name = os.getenv('ENGINE_LOGGING_LEVEL', 'INFO')
_default_level = logging.getLevelName(_default_level_name.upper())


# ----------------------------------------------------------------------
# 集成的辅助函数：确保目录存在
# ----------------------------------------------------------------------
def ensure_dir(path):
    """
    检查目录是否存在，如果不存在则创建。
    这个函数取代了 pyt_utils.ensure_dir，使 logger.py 不再有外部依赖。
    """
    if not os.path.exists(path):
        os.makedirs(path)


class LogFormatter(logging.Formatter):
    log_fout = None
    date_full = '[%(asctime)s %(lineno)d@%(filename)s:%(name)s] '
    date = '%(asctime)s '
    msg = '%(message)s'

    def format(self, record):
        if record.levelno == logging.DEBUG:
            mcl, mtxt = self._color_dbg, 'DBG'
        elif record.levelno == logging.WARNING:
            mcl, mtxt = self._color_warn, 'WRN'
        elif record.levelno == logging.ERROR:
            mcl, mtxt = self._color_err, 'ERR'
        else:
            mcl, mtxt = self._color_normal, ''

        if mtxt:
            mtxt += ' '

        if self.log_fout:
            self.__set_fmt(self.date_full + mtxt + self.msg)
            formatted = super(LogFormatter, self).format(record)
            # 文件输出部分被注释掉，但格式化逻辑保留
            return formatted

        self.__set_fmt(self._color_date(self.date) + mcl(mtxt + self.msg))
        formatted = super(LogFormatter, self).format(record)

        return formatted

    if sys.version_info.major < 3:
        def __set_fmt(self, fmt):
            self._fmt = fmt
    else:
        def __set_fmt(self, fmt):
            self._style._fmt = fmt

    @staticmethod
    def _color_dbg(msg):
        return '\x1b[36m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_warn(msg):
        return '\x1b[1;31m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_err(msg):
        return '\x1b[1;4;31m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_omitted(msg):
        return '\x1b[35m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_normal(msg):
        return msg

    @staticmethod
    def _color_date(msg):
        return '\x1b[32m{}\x1b[0m'.format(msg)


def get_logger(log_dir=None, log_file=None, formatter=LogFormatter):
    logger = logging.getLogger()
    logger.setLevel(_default_level)
    # 清除旧的 handlers，避免重复输出
    del logger.handlers[:] 

    if log_dir and log_file:
        # 使用集成的 ensure_dir 函数
        ensure_dir(log_dir) 
        LogFormatter.log_fout = True
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter(datefmt='%d %H:%M:%S'))
    stream_handler.setLevel(0)
    logger.addHandler(stream_handler)
    return logger