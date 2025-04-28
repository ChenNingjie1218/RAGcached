from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from llama_index.core.llms import CustomLLM, CompletionResponseGen, CompletionResponse
import time
import torch
from src.configs.config import model_path, kv_cache_output_dir
from typing import Any
from pydantic import Field
from transformers.cache_utils import (
    DynamicCache,
)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class LocalStreamingLLM(CustomLLM):    
    model: Any = Field(default=None, description="HuggingFace模型")
    tokenizer: Any = Field(default=None, description="HuggingFace分词器")
    streamer: Any = Field(default=None, description="文本流生成器")
    prefix_cache: Any = Field(default=None, description="PREFIX的KV cache")
    PREFIX: Any = Field(default=None, description="PREFIX")
    
    def __init__(self):
        """
        初始化本地流式语言模型
        """
        try:
            super().__init__()
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
            
            # 预计算PREFIX的KV cache
            self.PREFIX = '''<|im_start|>system
你是一个准确可靠的AI助手，能够借助外部文档回答问题。如果文档中包含正确答案，你会给出准确的回答。如果文档中没有包含答案，你会生成"由于文档中信息不足，我无法回答这个问题。"<|im_end|><|im_start|>user\n文档:
[ 2023-07-28 16：40 ] ，正文：人民网柏林7月28日电 （记者刘仲华、张慧中）7月27日晚，中国驻德国大使馆举行庆祝中国人民解放军建军96周年招待会。德军政官员、各界人士、多国驻德使节和武官、华侨华人、留学生及中资企业代表等约350人出席招待会。吴恳大使和使馆主要外交官出席活动。国防武官吴俊辉少将在招待会上致辞。吴俊辉强调，中国人民解放军建军96周年来，在中国共产党领导下，为民族独立和解放，为国家建设和发展，为维护国家主权、安全和发展利益，建立了不朽功勋。中国军队永远是捍卫国家主权和领土完整的坚定力量。吴俊辉针对当前涉中国问题的负面杂音，简要阐述了中国和平理念和国防政策，指出历史是最好的见证者，中国是全球唯一将“坚持和平发展道路”载入宪法的国家，在涉及和平与安全问题上，中国更是纪录最好的大国。中国从不追逐霸权，也绝不走“国强必霸”的老路，中国军队始终是维护世界和平的坚定力量。吴俊辉表示，李强总理上月对德国进行了正式访问，传承了两国友谊，深化了双边合作。今年初，中德两国国防部举行了工作对话，前不久，两国防长在香格里拉对话会期间进行了会晤，为两军关系下一步发展指明了方向。中国军队愿同德方一道，加强沟通交流，增进理解互信，深化务实合作，共同维护世界和地区安全与稳定。招待会气氛热烈友好，现场还通过电子屏幕滚动播放《我在》等视频。出席来宾纷纷祝贺近年来中国国防和军队建设取得的巨大成就，赞赏中国军队为维护世界和平所做出的积极贡献。分享让更多人看到
2023-08-11 15:10，正文：中新网约翰内斯堡8月10日电 (记者 王曦)“中国-南非企业贸易对接会暨签约仪式”10日在南非约翰内斯堡举行。中国商务部部长王文涛、南非贸工部部长易卜拉欣·帕特尔、中国驻南非大使陈晓东、南非驻华大使谢胜文、金砖国家工商理事会南非理事会理事斯塔夫罗斯·尼古拉，以及南非贸工部高级官员和来自中南两国企业家代表280多人参加活动。王文涛表示，近年来，两国元首保持战略沟通，引领推动双边经贸合作取得长足发展。中南双边贸易稳步提升，双向投资规模扩大。当前，中国正在以高质量发展推进中国式现代化，将为包括南非在内的世界各国带来重要机遇，中南、中非经贸合作前景光明、大有可为。今年是中南建交25周年，也是金砖“南非年”。中方组织贸易促进团来南非开展进口采购，是落实两国元首共识的实际行动。愿通过此次活动，进一步挖掘双方合作潜力，拓展合作空间。中方坚持对外开放基本国策，稳步扩大制度型开放，不断提升贸易投资自由化便利化水平，愿继续积极扩大自各方进口。诚挚欢迎南非企业来华参加第六届进博会等展会，加大对南非优势特色产品的推介力度，进一步开拓中国市场。易卜拉欣·帕特尔表示，南中两国建交以来，双边经贸合作取得诸多务实成果，为两国关系进一步发展打下良好基础。南非政府支持中南企业加强合作，将始终为中国企业来南开展贸易和投资打开大门，继续推进南中贸易投资等领域合作，促进双边经贸关系更好发展。陈晓东表示，近年来，中南两国经贸合作持续健康发展。金砖国家领导人第15次会晤将为中南两国及金砖国家发展带来新的重要机遇。希望中南商界充分利用贸易对接会平台，不断推动中南经贸合作扩容增量、提质增效。尼古拉表示，此次贸易对接会对于推动南中经贸领域更深入合作具有重要作用。希望南非企业抓住这一机会，让更多南非农产品、医药产品、矿产品等出口到中国。此次活动由中国商务部外贸发展事务局承办。中国保利集团、中国诚通集团、中国农业发展集团、比亚迪集团、宁德时代新能源科技股份有限公司等20家中国企业参加活动。对接会上，双方企业签署了多项贸易协议。(完)
2023-08-17 17:40，正文：去年，华硕无畏与英国艺术家Philip Colbert合作，将超人气IP龙虾人与无畏二合一笔记本结合，推出艺术家联名深度定制款，凭借张扬生动、极富色彩冲击力的形象打入Z世代，也自此开启了品牌的IP联动之旅。8月华硕无畏官宣，将携手国际街头潮流服饰品牌A BATHING APE®（BAPE®），突破时尚边界，融合前卫创意，打造一款同时具备卓越性能和时尚潮流的数码产品。15日，华硕笔记本官方微博发布#华硕无畏BAPE联名限定版#笔记本电脑礼盒预告视频，透露礼盒中将含有笔记本电脑、鼠标、公仔、电脑包等超多联名周边，将为年轻潮流爱好者们带来个性十足的设计佳作。华硕无畏是华硕旗下的轻薄笔记本系列，产品线丰富，分为“有标压，更高能”的【华硕无畏系列】和“有独显，更全能”的【华硕无畏Pro系列】，广泛适用于学习、办公、娱乐和内容创作等各类场景，深受现代年轻用户的喜爱。今年，华硕无畏推出了多款采用第13代英特尔酷睿和锐龙7000系列处理器的轻薄笔记本新品，包括无畏15i 2023、无畏15 2023、无畏16 2023、无畏Pro15 2023和无畏Pro16 2023，凭借出色的性能再次收获了诸多好评。BAPE®作为潮流界的巅峰代表、街头时尚的指标，独特的街头风格和标志性的迷彩图案征服了无数潮人的心。联名合作限定版LOGO、BABY MILO®代表性形象加上华硕无畏的高性能配置，设计和技术的完美结合让华硕无畏BAPE®联名限定版成为绝对的“潮物”。惊喜即将揭晓，想了解更多关于华硕无畏BAPE®联名限定版笔记本电脑礼盒的信息，请关注华硕笔记本官方微博，获取最新的产品介绍、活动信息和购买方式，一起期待这款潮流科技新品的发布！
08/16  13:50，正文：参考消息网8月16日报道 据《今日美国报》网站8月14日报道，美国夏威夷州州长乔希·格林说，毛伊岛小镇拉海纳大火致死人数可能增至目前的两倍甚至三倍。这场灾难截至目前导致至少99人丧生，已经成为一个多世纪以来美国最致命的野火。库利特表示，事发时俄军战机正按预定计划执行任务，在阿坦夫地区9100米高度沿叙利亚南部边境飞行。两架隶属“国际联盟”的F-35隐形战机于当日12:35到12:50危险接近俄军战机，持续时间约为15分钟。库利特15日还说，“国际联盟”战机在过去一天内20次违反消除冲突协议，俄方发现并记录下了“国际联盟”6架F-16战机、6架F-35隐形战机、4架“阵风”战机、两架“台风”战机以及两架MQ-1C无人机的违规行为。另外，“国际联盟”的无人机也实施了9次违规行为。此前，据“保加利亚军事网”7月22日报道，俄罗斯驻叙利亚冲突各方调解中心还曾声称，一架美空军F-16战斗机曾在叙利亚南部边境地区开启火控系统，锁定正在按计划执行预定任务的俄空天军战机。“保加利亚军事网”报道称，美军战机当时已经准备对俄罗斯战机发射武器，俄军机装备的示警系统记录下了美军F-16战机的行为。俄方还表示，西方无人机多次在未与俄方协调的情况下，非法接近位于叙利亚的俄罗斯军事设施。俄罗斯不会对这些西方军事资产的安全负责，因为西方未向俄方通报这些无人机的飞行活动。
2023-07-29 11:49:25，正文：        Sayings：许多女孩，都曾经梦想拥有一个芭比娃娃。这只漂亮、纤细、似乎无所不能的娃娃，就是理想的、长大后的样子。“我也要像她那样完美！”女孩们说。于是，芭比，本是理想的寄托，却不知不觉成为了枷锁……                                                “芭比”是完美的，也是虚幻的。现实世界不是Barbieland，你也无需成为完美的“芭比”。                                【写在最后】“你必须瘦，又不能太瘦。你不能说自己想瘦，你得说，你是为了健康，所以不得不逼着自己瘦。你要有钱，但是不能张口要钱，否则就是俗。你要往上爬，但不能靠手腕。要有领导力，但不能压制别人的想法。……你永远不能变老， 永远不能失态；永远不能炫耀；永远不能自私；永远不能消沉；不能失败；不能胆怯；永远不能离经叛道。这太困难了，处处都是矛盾，而且绝对不会有人奖励你或感谢。你到了最后，你不但做错了所有事，而且所有的错都怪在你头上。”电影《芭比》是的，做一个“永远正确的女性”，是不可能完成的任务。所以，去他的。你不必活成别人的期待，也不必完美，最重要的是，你是你自己。设计：葵子、土白白责编：des        “It is literally impossible to be a woman.”“做女人是不可能完成的任务。”——《芭比》'''
            prefix_inputs = self.tokenizer(self.PREFIX, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                self.prefix_cache = DynamicCache()
                self.prefix_cache = self.model(**prefix_inputs, past_key_values = self.prefix_cache).past_key_values
                # kvcache_file_path = f'{kv_cache_output_dir}/kvcache_chunk.pt'
                # torch.save(self.prefix_cache, kvcache_file_path)
            
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")
    
    @property
    def metadata(self):
        """返回模型元数据"""
        return {
            "model_name": "Qwen2.5-7B",
            "context_window": 32768,  # Qwen2.5-7B 的上下文窗口大小
            "num_output": 1280,  # 与 max_new_tokens 保持一致
        }
    
    def stream_complete(self, prompt: str) -> CompletionResponseGen:
        """
        流式生成文本
        
        Args:
            prompt: 输入提示词
            
        Yields:
            CompletionResponse: 生成的文本响应
        """
        try:
            # 在生成开始前就开始计时
            start_time = time.time()
            first_token_received = False

            inputs = self.tokenizer(self.PREFIX+prompt, return_tensors="pt").to(self.model.device)
            kvcache_file_path = f'{kv_cache_output_dir}/kvcache_chunk.pt'
            past_key_values = torch.load(kvcache_file_path, weights_only=False)
            # past_key_values = copy.deepcopy(self.prefix_cache)
            generation_kwargs = {
                **inputs,
                "max_new_tokens": 512,
                "streamer": self.streamer,
                "past_key_values": past_key_values,
            }
            
            # 使用多线程进行生成
            import threading
            thread = threading.Thread(target=self._safe_generate, kwargs=generation_kwargs)
            thread.start()
            
            # 逐Token捕获输出
            for token in self.streamer:
                if not first_token_received:
                    ttft = time.time() - start_time  # 计算TTFT
                    first_token_received = True
                    print(f"TTFT: {ttft:.2f}s")
                yield CompletionResponse(text=token, delta=token)
                
        except Exception as e:
            raise RuntimeError(f"文本生成失败: {str(e)}")
    
    def complete(self, prompt: str) -> str:
        """
        非流式生成文本
        
        Args:
            prompt: 输入提示词
            
        Returns:
            str: 生成的完整文本
        """
        try:
            inputs = self.tokenizer(self.PREFIX+prompt, return_tensors="pt").to(self.model.device)
            kvcache_file_path = f'{kv_cache_output_dir}/kvcache_chunk.pt'
            past_key_values = torch.load(kvcache_file_path, weights_only=False)
            generation_kwargs = {
                **inputs,
                "max_new_tokens": 512,
                "past_key_values": past_key_values,
            }
            
            outputs = self._safe_generate(**generation_kwargs)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 移除输入提示词，只返回生成的文本
            return response[len(prompt):]
            
        except Exception as e:
            raise RuntimeError(f"文本生成失败: {str(e)}")
        
    def _safe_generate(self, **kwargs):
        with torch.no_grad():
            return self.model.generate(**kwargs)
    
