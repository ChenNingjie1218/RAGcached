from src.configs.config import MAX_TRIE_NODE_COUNT
from src.logger.performance_logger import PerformanceLogger

def singleton(cls):
    """装饰器：用于将类变为单例"""
    instances = {}
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper

class TrieNode:
    """Trie节点类
    存储kv cache
    """
    def __init__(self, kv, node_id, token_count):
        self.children = {}  # 子节点字典
        self.kv = kv
        self.id = node_id
        self.token_count = token_count
        self.frequency = 0

    def increase_frequency(self):
        self.frequency += 1

@singleton
class KVCacheTrie:
    def __init__(self, prefix_kv = None, prefix_token_count = 0):
        self.root = TrieNode(kv=prefix_kv, node_id=0, token_count=prefix_token_count)  # 根节点存 prefix kv 值
        self.node_count = 1  # 节点总数
        self.token_count = prefix_token_count

    def insert(self, node_list):
        """插入一个 KV Cache 列表作为路径"""
        current = self.root
        for node_info in node_list:
            node_id = node_info['id']
            node_kv = node_info['kv']
            node_token_count = node_info['token_count']

            if node_id not in current.children:
                new_node = TrieNode(kv=node_kv, node_id=node_id, token_count=node_token_count)
                self.node_count += 1
                self.token_count += node_token_count
                print(f"插入新节点，当前节点数：{self.node_count}，当前token数: {self.token_count}")
                current.children[node_id] = new_node
            current = current.children[node_id]

    def search_longest_path(self, node_set):
        """搜索最长路径，路径中的每个节点的 KV Cache 必须存在于集合中，返回最长路径的节点 ID 列表"""
        
        max_path = []
        current_path = [self.root]
        self._dfs(self.root, current_path, node_set, max_path)

        # 更新路径上所有节点的访问频率
        for node in max_path:
            node.increase_frequency()
        print(f'复用节点数：{len(max_path)}')
        PerformanceLogger.record_event("kvcachetrie", "reuse", {"reuse":len(max_path)})
        return max_path

    def _dfs(self, node, current_path, node_set, max_path):
        # 更新最长路径
        if len(current_path) > len(max_path):
            max_path[:] = current_path[:]

        # 遍历当前节点的所有子节点
        for child in node.children.values():
            if child.id in node_set:
                current_path.append(child)
                self._dfs(child, current_path, node_set, max_path)
                current_path.pop()
