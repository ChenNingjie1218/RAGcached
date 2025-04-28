import logging
import time
import torch 
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llms.local_model import LocalStreamingLLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

llm = LocalStreamingLLM()

# prompt = """
# [ 2023-07-28 16：40 ] ，正文：人民网柏林7月28日电 （记者刘仲华、张慧中）7月27日晚，中国驻德国大使馆举行庆祝中国人民解放军建军96周年招待会。德军政官员、各界人士、多国驻德使节和武官、华侨华人、留学生及中资企业代表等约350人出席招待会。吴恳大使和使馆主要外交官出席活动。国防武官吴俊辉少将在招待会上致辞。吴俊辉强调，中国人民解放军建军96周年来，在中国共产党领导下，为民族独立和解放，为国家建设和发展，为维护国家主权、安全和发展利益，建立了不朽功勋。中国军队永远是捍卫国家主权和领土完整的坚定力量。吴俊辉针对当前涉中国问题的负面杂音，简要阐述了中国和平理念和国防政策，指出历史是最好的见证者，中国是全球唯一将“坚持和平发展道路”载入宪法的国家，在涉及和平与安全问题上，中国更是纪录最好的大国。中国从不追逐霸权，也绝不走“国强必霸”的老路，中国军队始终是维护世界和平的坚定力量。吴俊辉表示，李强总理上月对德国进行了正式访问，传承了两国友谊，深化了双边合作。今年初，中德两国国防部举行了工作对话，前不久，两国防长在香格里拉对话会期间进行了会晤，为两军关系下一步发展指明了方向。中国军队愿同德方一道，加强沟通交流，增进理解互信，深化务实合作，共同维护世界和地区安全与稳定。招待会气氛热烈友好，现场还通过电子屏幕滚动播放《我在》等视频。出席来宾纷纷祝贺近年来中国国防和军队建设取得的巨大成就，赞赏中国军队为维护世界和平所做出的积极贡献。分享让更多人看到
# 问题：
# 庆祝中国人民解放军建军96周年招待会是什么时候举行的？<|im_end|>
# <|im_start|>assistant\n"""

prompt = """
问题：
庆祝中国人民解放军建军96周年招待会是什么时候举行的？<|im_end|>
<|im_start|>assistant\n"""

# prompt = """动物园的老虎叫花花.
# 问题：
# 动物园的老虎叫什么？<|im_end|>
# <|im_start|>assistant\n"""

# 流式生成
for response in llm.stream_complete(prompt):
    print(response.text, end="", flush=True)
# # 非流式生成
# response = llm.complete(prompt)
# print(response)
