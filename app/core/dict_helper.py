def touch_cache_key(_agent_chain_cache, key: tuple):
    """
    模拟 OrderedDict.move_to_end(key) 的行为。
    将指定键的元素移动到字典的末尾，表示它最近被使用。
    """
    if key in _agent_chain_cache:
        value = _agent_chain_cache.pop(key)  # 删除该键值对
        _agent_chain_cache[key] = value  # 重新插入，使其成为最后一个


def pop_lru_item(_agent_chain_cache):
    """
    模拟 OrderedDict.popitem(last=False) 的行为。
    删除并返回字典中第一个（最旧的）键值对。
    """
    if not _agent_chain_cache:
        return None  # 或者抛出 KeyError，取决于你的需求

    # 获取字典中的第一个键（即最旧的键）
    lru_key = next(iter(_agent_chain_cache))
    lru_value = _agent_chain_cache.pop(lru_key)  # 删除并返回
    return lru_key, lru_value