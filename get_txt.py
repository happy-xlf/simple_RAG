#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   get_txt.py
@Time    :   2024/05/05 14:41:49
@Author  :   Lifeng
@Version :   1.0
@Desc    :   None
'''
from newspaper import Article
from transformers import AutoTokenizer, AutoModel
import os,json

# 网页文章解析
def get_article_text(url):
    a = Article(url)
    try:
        a.download()
        a.parse()
        print(a.text)
        return a.text.replace("\n\n","\n").replace("\n\n\n\n","")
    except Exception as e:
        print(f"url解析失败，错误原因：{e}")
        return ""

# url = "https://zhuanlan.zhihu.com/p/638426349"
# url = "https://zhuanlan.zhihu.com/p/42252563"
url = "https://zhuanlan.zhihu.com/p/368442411"

content = get_article_text(url)

with open("./rag_documents/tianjin.txt", "w", encoding="utf-8") as f:
    f.write(content)