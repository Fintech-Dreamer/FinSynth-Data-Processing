import os
import time
import warnings
import json
from typing import Literal

import pandas as pd

from utils import (
    pdf_to_json,
    html_to_json,
    txt_md_to_json,
    doc_docx_to_json,
    ppt_pptx_to_json,
    image_to_csv,
    xml_to_json,
    wav_to_json,
    generate_questions_on_chatbot,
    generate_questions_on_fraud,
    generate_questions_on_compliance,
)
from params import API_KEY, API_BASE, MODEL, MODEL_PICTURE

warnings.filterwarnings("ignore")


class FileConverter:
    """文件转换类
    paths:读取文件路径列表
    支持格式:pdf, html, csv
    例:
        file_converter = FileConverter(["base.csv","Attention Is All You Need.pdf"])
    或
        file_converter = FileConverter(["fraud.csv"])
    注意:
        html推荐变成带文字的pdf格式再进行处理(比如让大模型帮你生成md格式再转化pdf),个人认为是有反爬导致的无法转化html
    """

    def __init__(self, paths: list):
        self.paths = paths
        self.json_lists = []
        self.QA_pairs = []

    def file_to_json_lists(self):
        """将文件转换为json list"""
        if self.paths == []:
            print("No file path provided.")
            return None
        for path in self.paths:
            print(f"Converting {path} to json...")
            _, ext = os.path.splitext(path)
            if ext.lower() == ".pdf":
                self.json_lists.append(pdf_to_json(path))
            elif ext.lower() == ".html" or "http" in path:
                self.json_lists.append(html_to_json(path))
            elif ext.lower() == "txt" and ext.lower() == "md":
                self.json_lists.append(txt_md_to_json(path))
            elif ext.lower() == ".doc" or ext.lower() == ".docx":
                self.json_lists.append(doc_docx_to_json(path))
            elif ext.lower() == ".ppt" or ext.lower() == ".pptx":
                self.json_lists.append(ppt_pptx_to_json(path))
            elif ext.lower() == ".jpg" or ext.lower() == ".png" or ext.lower() == ".jpeg":
                self.json_lists.append(image_to_csv(path))
            elif ext.lower() == ".xml":
                self.json_lists.append(xml_to_json(path))
            elif ext.lower() == ".wav":
                self.json_lists.append(wav_to_json(path))
            else:
                print(f"Unsupported file type: {path}")
                self.json_lists.append(None)
        return self.json_lists

    def json_lists_to_QA_pairs(self, choice: Literal["chatbot", "fraud", "compliance"], time_sleep: int = 60, lable: int = -1, embed_model_name: str = "BAAI/bge-m3"):
        """将json list转换为问答对
        choice:选择生成问答对的类型,可选值:chatbot, fraud, compliance
        time_sleep:生成问答对间隔时间
        lable:选择json list的前几个元素,默认为-1,即全部元素
        embed_model_name:选择embeddings的模型,默认为BAAI/bge-m3
        """
        if self.json_lists == []:
            print("please run file_to_json_lists() first")
            return None
        if choice not in ["chatbot", "fraud", "compliance"]:
            print("Invalid choice,please choose from 'chatbot', 'fraud', 'compliance'")
            return None
        if choice == "chatbot":
            for i, json_list in enumerate(self.json_lists):
                print(f"Converting {self.paths[i]} elements to QA pairs...")
                self.QA_pairs.extend(generate_questions_on_chatbot(json_list, API_KEY, API_BASE, MODEL, MODEL_PICTURE, lable=lable, embed_model_name=embed_model_name))
                print(self.QA_pairs[-1])
                print("-" * 50)
                time.sleep(time_sleep)
        if choice == "fraud":
            for i, json_list in enumerate(self.json_lists):
                for flag, element in enumerate(json_list):
                    print(f"Converting {self.paths[i]} element {flag + 1} to QA pairs...")
                    self.QA_pairs.append(generate_questions_on_fraud(element, API_KEY, API_BASE, MODEL))
                    print(self.QA_pairs[-1])
                    print("-" * 50)
                    time.sleep(time_sleep)
        if choice == "compliance":
            for i, json_list in enumerate(self.json_lists):
                for flag, element in enumerate(json_list):
                    print(f"Converting {self.paths[i]} element {flag + 1} to QA pairs...")
                    self.QA_pairs.extend(generate_questions_on_compliance(element, API_KEY, API_BASE, MODEL))
                    print(self.QA_pairs[-2], "\n", self.QA_pairs[-1])
                    print("-" * 50)
                    time.sleep(time_sleep)

    def read_json_lists(self, filename: str):
        """读取json list文件
        filename:json list文件名
        """
        self.paths = ["noname"]
        with open(filename, "r", encoding="utf-8") as f:
            self.json_lists.append(json.load(f))

    def save_json_lists(self, filename: str = "output.json"):
        """保存json list文件为json格式
        filename:json list文件名
        """
        if self.json_lists == []:
            print("please run file_to_json_lists() first")
            return None
        for i, json_list in enumerate(self.json_lists):
            with open(f"{filename}_{i + 1}.json", "w", encoding="utf-8") as f:
                json.dump(json_list, f, indent=2, ensure_ascii=False)

    def save_QA_pairs(self, filename: str = "output_QA_pairs.csv"):
        """保存问答对文件为csv格式
        filename:问答对文件名
        """
        if self.QA_pairs == []:
            print("please run json_lists_to_QA_pairs() first")
            return None
        QA_pairs_csv = pd.DataFrame(self.QA_pairs)
        QA_pairs_csv.to_csv(filename, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    file_converter = FileConverter(["a.wav"])
    file_converter.file_to_json_lists()
    file_converter.save_json_lists()
    # file_converter.read_json_lists("output.json_1.json")
    # file_converter.json_lists_to_QA_pairs("chatbot", time_sleep=0, lable=10, embed_model_name="BAAI/bge-m3")
    # file_converter.save_QA_pairs()
