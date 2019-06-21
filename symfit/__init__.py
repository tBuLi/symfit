from symfit.api import *
import pkg_resources

try:
    __version__ = pkg_resources.get_distribution('symfit').version
except pkg_resources.DistributionNotFound:
    __version__ = ''
