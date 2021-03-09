def subclasses(base, leaves_only=True):
    """
    Recursively create a set of subclasses of ``object``.

    :param object: Class
    :param leaves_only: If ``True``, return only the leaves of the subclass tree
    :return: (All leaves of) the subclass tree.
    """
    base_subs = set(base.__subclasses__())
    if not base_subs or not leaves_only:
        all_subs = {base}
    else:
        all_subs = set()
    for sub in list(base_subs):
        sub_subs = subclasses(sub, leaves_only=leaves_only)
        all_subs.update(sub_subs)
    return all_subs