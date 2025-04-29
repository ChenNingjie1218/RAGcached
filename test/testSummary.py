import logging
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llms.api_model import API_LLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

llm = API_LLM()

query  = '''人民网柏林7月28日电 中国驻德国大使馆27日晚举办建军96周年招待会，德军政界、多国使节及各界代表约350人出席。国防武官吴俊辉少将强调，中国军队在中国共产党领导下为国家发展和安全作出重大贡献，是捍卫国家主权的坚定力量。他重申中国将"和平发展"写入宪法，坚持永不称霸，致力于维护世界和平。吴俊辉提及李强总理访德成果及中德国防部近期对话，表示愿深化两军交流合作。活动期间，来宾高度评价中国国防建设成就及维和贡献，现场播放相关主题视频。'''

response = llm.complete(query)

logger.info(f"压缩后的文本为:\n {response}")
