# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

import importlib
import pkgutil
import sys

def _import_submodules(package_name):
    """ Import all submodules of a module, recursively
    Adapted from: http://stackoverflow.com/a/25083161

    :param package_name: Package name
    :type package_name: str
    :rtype: dict[types.ModuleType]
    """
    package = sys.modules[package_name]
    out = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        try:
            #module = importlib.import_module('{}.{}'.format(package_name, name))
            module = importlib.import_module('.{}'.format(name), package=package_name)
            out[name] = module
        except:
            continue
    return out
_submodules = _import_submodules(__name__)
__all__ = list(_submodules.keys())

