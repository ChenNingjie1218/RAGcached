'''
压缩上下文的prompt前缀
'''


COMPRESS_PREFIX = '''<|im_start|>system
            请对以下文本进行去冗余处理，保留关键信息。只需要返回一个版本，且是完整的自然段落形式，不要重复输出、不要分段、不要编号、不要加标题。<|im_end|>
            <|im_start|>user
            以下是原文档内容：
            '''