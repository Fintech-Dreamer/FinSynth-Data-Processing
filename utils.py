import io
import os
import warnings
import base64


# ========== 音频处理模块 ==========
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tempfile import mkdtemp

# ========== 文档解析模块 ==========
from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text
from unstructured.partition.doc import partition_doc
from unstructured.partition.docx import partition_docx
from unstructured.partition.ppt import partition_ppt
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.image import partition_image
from unstructured.partition.xml import partition_xml

# ========== 数据结构与验证模块 ==========
from pydantic import BaseModel, Field
from typing import List

# ========== 大模型应用模块 ==========
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# ========== 向量数据库模块 ==========
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# ========== HTML/XML处理模块 ==========
from bs4 import BeautifulSoup

# ========== 图像处理模块 ==========
from PIL import Image


warnings.filterwarnings("ignore")


def pdf_to_json(file_path: str, min_words: int = 20) -> list:
    """将pdf文件转换为json list:
    file_path:pdf文件路径
    min_words:筛选掉词数小于min_words的元素
    返回json list
    """
    try:
        elements = partition_pdf(
            file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            split_pdf_page=True,
            split_pdf_allow_failed=True,
            split_pdf_concurrency_level=15,
        )
        output_list = [element.to_dict() for element in elements]
        output_list_modified = tables_from_html(output_list)
        output_list_modified = filter_words(output_list_modified, min_words=min_words)
        return output_list_modified
    except KeyError as e:
        print(e)
        return e.message


def html_to_json(path: str) -> list:
    """将html文件转换为json list:
    file_path:html文件路径
    min_words:筛选掉词数小于min_words的元素
    返回json list
    """
    try:
        if "http" in path:
            elements = partition_html(url=path)
        else:
            elements = partition_html(filename=path)
        output_list = [element.to_dict() for element in elements]
        return output_list
    except KeyError as e:
        print(e)
        return e.message


def txt_md_to_json(file_path: str) -> list:
    try:
        elements = partition_text(file_path)
        output_list = [element.to_dict() for element in elements]
        return output_list
    except KeyError as e:
        print(e)
        return e.message


def doc_docx_to_json(file_path: str) -> list:
    try:
        if file_path.endswith(".doc"):
            elements = partition_doc(file_path)
        elif file_path.endswith(".docx"):
            elements = partition_docx(file_path)
        output_list = [element.to_dict() for element in elements]
        return output_list
    except KeyError as e:
        print(e)
        return e.message


def ppt_pptx_to_json(file_path: str) -> list:
    try:
        if file_path.endswith(".ppt"):
            elements = partition_ppt(file_path)
        elif file_path.endswith(".pptx"):
            elements = partition_pptx(file_path)
        output_list = [element.to_dict() for element in elements]
        return output_list
    except KeyError as e:
        print(e)
        return e.message


def image_to_csv(file_path: str) -> list:
    try:
        elements = partition_image(file_path)
        output_list = [element.to_dict() for element in elements]
        return output_list
    except KeyError as e:
        print(e)
        return e.message


def xml_to_json(file_path: str) -> list:
    try:
        elements = partition_xml(file_path)
        output_list = [element.to_dict() for element in elements]
        return output_list
    except KeyError as e:
        print(e)
        return e.message


def split_audio_chunks(audio_path, chunk_sec=10):
    """将音频分割为固定时长的块（单位：秒）"""
    audio = AudioSegment.from_wav(audio_path)
    chunk_length = chunk_sec * 1000  # 转换为毫秒
    return [audio[i : i + chunk_length] for i in range(0, len(audio), chunk_length)]


def audio_to_json(audio_path, language="zh-CN", silence_thresh=-40, min_silence_len=500, keep_silence=300, debug_mode=False):
    """
    将音频文件转换为带时间戳的结构化文本

    参数说明：
    :param audio_path: 音频文件路径（支持mp3/wav等格式）
    :param language: 识别语言（默认中文）
    :param silence_thresh: 静音阈值(dBFS)，越小越敏感（-50~-35）
    :param min_silence_len: 视为静音的最小持续时间（毫秒）
    :param keep_silence: 分块前后保留的静音时长（毫秒）
    :param debug_mode: 是否保留临时文件（默认False）
    :return: 包含时间戳和文本的结构化列表
    """
    # 初始化组件
    recognizer = sr.Recognizer()
    temp_dir = mkdtemp()
    results = []

    try:
        # 加载并统一音频格式
        audio = AudioSegment.from_file(audio_path).set_frame_rate(16000).set_channels(1)

        # 静音分块检测
        chunks = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=keep_silence)

        current_time = 0.0
        for idx, chunk in enumerate(chunks):
            # 计算时间戳
            chunk_duration = len(chunk) / 1000.0
            end_time = current_time + chunk_duration

            # 保存临时分块
            chunk_path = os.path.join(temp_dir, f"chunk_{idx}.wav")
            chunk.export(chunk_path, format="wav")

            # 语音识别
            try:
                with sr.AudioFile(chunk_path) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data, language=language, show_all=True)

                    if text and "alternative" in text:
                        best = text["alternative"][0]
                        results.append({"start": round(current_time, 2), "end": round(end_time, 2), "text": best["transcript"], "confidence": round(best.get("confidence", 0), 2)})
            except sr.UnknownValueError:
                print(f"无法识别 {current_time:.1f}s-{end_time:.1f}s 的音频")
            except sr.RequestError as e:
                print(f"API请求失败: {e}")

            current_time = end_time  # 更新时间指针

    finally:
        # 清理临时文件
        if not debug_mode and os.path.exists(temp_dir):
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)

    return results


def generate_questions_on_chatbot(json_list: list, openai_api_key: str, base_url: str, model: str, model_picture: str, lable: int, embed_model_name: str) -> list:
    try:
        if lable != -1:
            json_list_lable = json_list[:lable]
            embed_model = HuggingFaceBgeEmbeddings(model_name=embed_model_name)  # model_name存放embeddings的路径
            while len(json_list_lable) >= 10:
                # 合并10个元素块的文本和图片摘要，顺序融合，并且过滤较短的文本块
                json_lists_use = [block for block in json_list_lable if len(block["text"]) > 200]
                texts_4k_token = [x["text"] for x in json_lists_use]
                images_list_base64 = [x["metadata"]["image_base64"] if "image_base64" in x["metadata"] else "" for x in json_lists_use]
                images_list = [image_summarize(image_base64, openai_api_key, base_url, model_picture) if image_base64 else "" for image_base64 in images_list_base64]
                context_list = [str(texts_4k_token[i] + images_list[i]) for i in range(len(json_lists_use))]
                # context转成向量库
                vectorstore = vectorstore_create(context_list, embed_model)
                # 生成问题，每个元素块生成一个问题
                json_list_lable = json_list_lable[10:] if len(json_list_lable) > 10 else []
                questions_list = get_questions_only(texts_4k_token, openai_api_key, base_url, model)
                # 文本向量库检索s
                Q_B_list = []
                Q_B_list = [{"question": query, "docs_list": "\n".join(text_retriever(query, vectorstore))} for query in questions_list]
                # 生成答案
                answer_list = generate_answer_only(Q_B_list, openai_api_key, base_url, model)
                # 生成问题答案对
                QA_list = [{"Question": Q_B_list[i]["question"], "Answer": answer_list[i], "Background": Q_B_list[i]["docs_list"]} for i in range(len(Q_B_list))]
                # 删除向量库,避免出现重复
                vectorstore = delete_vectorstore(vectorstore)
        return QA_list
    except Exception as e:
        print(e)
        return None


def generate_questions_on_fraud(element: dict, openai_api_key: str, base_url: str, model: str) -> list:
    """欺诈检测生成问答对:
    element:json list
    openai_api_key:openai api key
    base_url:openai api base url
    model:openai model
    """
    prompt = f"{element}"
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant.Please output JSON string, do not output other irrelevant content
                        Based on the provided data or scenario, form a description-answer pair.
                        The description should be a specific description of the data or scenario, and the answer should be a judgment on whether it is fraud.
                        If it is a fraudulent act, the answer should be "The transaction is Fraudulent because..."
                        If it is a non-fraudulent act, the answer should be "Not Fraudulent"
                        for example:
                        "QA_description": "This transaction occurred on 2023-01-22 03:56:37. The amount of $8.3 was spent in the category 'grocery_pos' at the merchant 'fraud_Deckow-O'Conner' located in Port Patrick.",
                        "QA_answer": "The transaction is Fraudulent because it happened at 03:56 AM for a little amount far from the customer."
                        What needs to be emphasized is:
                        The description should not contain information about the variable "fraud".
                        """,
        },
        {"role": "user", "content": prompt},
    ]
    chat = ChatOpenAI(
        openai_api_key=openai_api_key,
        base_url=base_url,
        model=model,
        temperature=0,
    )

    class FinalResponse(BaseModel):
        QA_description: str = Field(description="the description of the question-answer pair")
        QA_answer: str = Field(description="the answer to the question-answer pair")

    try:
        structured_llm = chat.with_structured_output(FinalResponse)
        res = structured_llm.invoke(messages)
        return {
            "instruction": "Please determine if the following information is financial fraud, answer in English.",
            "input": res.QA_description,
            "output": res.QA_answer,
        }
    except Exception as e:
        print(e)
        return None


def generate_questions_on_compliance(element: list, openai_api_key: str, base_url: str, model: str) -> list:
    """合规检测生成问答对:
    element:json list
    openai_api_key:openai api key
    base_url:openai api base url
    model:openai model
    """
    prompt = f"{element}"
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant.Please output JSON string, do not output other irrelevant content
                        Based on the provided data or scenario,form two text-answer pair,use chinese to answer.
                        The text should be based on the regulation or scenario to generate a specific compliance or non-compliance scenario or behavior, and the answer should be the corresponding compliance or non-compliance.
                        If it is a compliance act, the answer should be "是"
                        If it is a non-compliance act, the answer should be "否"
                        for example:
                        "QA_text_compliance": "作为金融机构的一名员工，我应该主动学习并遵守公司制定的所有合规制度，并在日常工作中严格遵循这些规定，确保我的行为符合法律法规的要求。",
                        "QA_answer_compliance": "是"
                        "QA_text_non_compliance": "我怎样才能获取那些未公开的内幕信息，从而进行违规的内幕交易，让我能够获取不当的利益呢？例如，我可以与某个公司高层秘密联络，获取他们即将公布的重大消息。",
                        "QA_answer_non_compliance": "否"
                        What needs to be emphasized is:
                        You need to generate two text-answer pairs, the first being compliant scenario or behavior and the second being non-compliance scenario or behavior.
                        """,
        },
        {"role": "user", "content": prompt},
    ]
    chat = ChatOpenAI(
        openai_api_key=openai_api_key,
        base_url=base_url,
        model=model,
        temperature=0,
    )

    class FinalResponse(BaseModel):
        QA_text_compliance: str = Field(description="the text of the compliant question-answer pair")
        QA_answer_compliance: str = Field(description="the answer to the compliant question-answer pair")
        QA_text_non_compliance: str = Field(description="the text of the non-compliant question-answer pair")
        QA_answer_non_compliance: str = Field(description="the answer to the non-compliant question-answer pair")

    try:
        structured_llm = chat.with_structured_output(FinalResponse)
        res = structured_llm.invoke(messages)
        return [
            {
                "instruction": "你是一个金融合规检测的专家，你会接收到一段文本和两个潜在的分类选项，请输出文本内容的正确类型",
                "text": res.QA_text_compliance,
                "category": '["是","否"]',
                "answer": res.QA_answer_compliance,
            },
            {
                "instruction": "你是一个金融合规检测的专家，你会接收到一段文本和两个潜在的分类选项，请输出文本内容的正确类型",
                "text": res.QA_text_non_compliance,
                "category": '["是","否"]',
                "answer": res.QA_answer_non_compliance,
            },
        ]
    except Exception as e:
        print(e)
        return None


def base64_to_image(output_list: list, output_path: str):
    """将base64字符串转换为图片并保存到指定路径:
    resp:从unstructured_client库获取的json list
    output_path:输出路径
    """
    for element in output_list:
        if "image_base64" in element["metadata"]:
            image_data = base64.b64decode(element["metadata"]["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image.save(os.path.join(output_path, f"image_{element['element_id']}.png"))


def tables_from_html(output_list: list) -> list:
    """将属性为Table的元素的html数据转换为表格数据并添加到元素的text属性中:
    output_list:从unstructured_client库获取的json list
    返回json list
    """
    for element in output_list:
        if "Table" in element["type"]:
            html_data = element["metadata"]["text_as_html"]
            soup = BeautifulSoup(html_data, "html.parser")
            table_data = []
            rows = soup.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                cols = [col.text.strip() for col in cols]
                if cols:
                    table_data.append(cols)
            element["text"] = table_data
    return output_list


def filter_words(input_list: list, min_words: int) -> list:
    """用来删选dict
    input_list:从unstructured_client库获取的json list
    min_words:最少词数
    is_complete_dict():判断是否为复合要求的dict
    返回json list
    """

    def is_complete_dict(element: dict, min_words) -> bool:
        return not all(
            [
                isinstance(element["text"], str) and len(element["text"].replace(" ", "")) < min_words,
                element["type"] != "Image",
                element["type"] != "Table",
            ]
        )

    output_list = [element for element in input_list if is_complete_dict(element, min_words)]
    return output_list


def image_summarize(img_base64, openai_api_key: str, base_url: str, model_picture: str) -> str:  # jpg格式
    """Make image summary"""
    chat = ChatOpenAI(api_key=openai_api_key, base_url=base_url, model=model_picture, temperature=0)
    # 构造提示词
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content


def get_questions_only(texts_4k_token: list, openai_api_key: str, base_url: str, model: str) -> list:
    class Entities(BaseModel):
        """generated questions"""

        questions: List[str] = Field(
            ...,
            description="""The generated questions.""",
        )

    summarys = str(texts_4k_token)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. Generate a question for each element in the input list.
                preferably related to financial analysis,
                Format your response as a JSON object with a single key 'questions' containing a list of questions.
                Example: {{"questions": ["What is the main idea?", "What are the key points?"]}}""",
            ),
            (
                "human",
                "Input list: {summarys}",
            ),
        ]
    )
    llm = ChatOpenAI(openai_api_key=openai_api_key, base_url=base_url, model=model, temperature=0)
    entity_chain = prompt | llm.with_structured_output(Entities)
    question_lists = entity_chain.invoke({"summarys": summarys})
    return question_lists.questions


def vectorstore_create(context: list, embed_model) -> Chroma:
    text_splitter = CharacterTextSplitter(
        separator=".",  # 因为分割对象是英文文本
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    texts = ".".join(context)
    context_list = text_splitter.split_text(texts)
    # 加载vectorstore
    vectorstore = Chroma.from_texts(texts=context_list, embedding=embed_model, collection_name="BAAI")
    return vectorstore


def delete_vectorstore(vectorstore: Chroma):
    # 获取Chroma客户端并删除已存在的集合
    client = chromadb.Client()
    client.delete_collection("BAAI")  # 确保集合名称正确
    return vectorstore


# 检索支撑事实（后续可以加上上下两个块，来扩大检索范围）
def text_retriever(query: str, vectorstore) -> list[str]:
    # retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold":0.25})
    # docs = retriever.get_relevant_documents(query)
    # 利用个数来筛选
    docs = vectorstore.similarity_search(query, k=2)
    if docs == []:
        docs_list = []
    else:
        docs_list = list(x.page_content for x in docs)
    return docs_list


def generate_answer_only(Q_B_list: list[str, str], openai_api_key: str, base_url: str, model: str) -> list[str, str]:
    prompt_list = [
        f"""Give me the simple answer based on the question and background 
        Question:{Q_B["question"]}
        Background:{Q_B["docs_list"]}
        Notice:Your output format must be answer:
        """
        for Q_B in Q_B_list
    ]

    chat = ChatOpenAI(openai_api_key=openai_api_key, base_url=base_url, model=model, temperature=0)
    messages_list = [
        [{"role": "system", "content": "You are a helpful assistant. Generate the Answer based on the Question and  Background"}, {"role": "user", "content": prompt}]
        for prompt in prompt_list
    ]

    # 调用 batch 方法并设置并发数
    responses = chat.batch(messages_list, config={"max_concurrency": 5})
    return [response.content for response in responses]


if __name__ == "__main__":
    a = "a.wav"
    print(a)
