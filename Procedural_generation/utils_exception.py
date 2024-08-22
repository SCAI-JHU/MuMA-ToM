
import tblib.pickling_support
tblib.pickling_support.install()
import sys


class UnityException(Exception):
    pass


class ManyFailureException(Exception):
    pass

class PlannerException(Exception):
    pass

class ExceptionWrapper(object):

    def __init__(self, ee):
        self.ee = ee
        __, __, self.tb = sys.exc_info()

    def re_raise(self):
        raise self.ee.with_traceback(self.tb)
        # for Python 2 replace the previous line by:
        # raise self.ee, None, self.tb