import xml.sax
import sys
from collections import defaultdict
import pkuseg


class Doc:
    def __init__(self, origlang, id, domain):
        self.origlang = origlang
        self.id = id
        self.domain = domain

    def __repr__(self):
        return "{0:20}{1:<5}{2:15}".format(self.origlang, self.id, self.domain)


class Sentencepair:
    def __init__(self, source=None, translate=None):
        self.source = source if source else []
        self.translate = translate if translate else defaultdict(list)

    # def __add__(self, other):
    #     source = self.source + other.source
    #     translate = {}
    #     for key in other.translate.keys():
    #         translate[key] = self.translate[key] + other.translate[key]
    #     return Sentencepair(source, translate)


sentencepair = Sentencepair()


class GetStorehouse(xml.sax.ContentHandler):  # 事件处理器
    def __init__(self):
        self.currentDoc = None
        self.state = None
        self.content = None

    def startElement(self, label, attr):  # 遇到元素开始标签出发该函数
        if label == "doc":  # 遇到一段文本
            if attr.get('domain'):
                self.currentDoc = Doc(origlang=attr['origlang'], id=attr['id'], domain=attr['domain'])
            else:
                self.currentDoc = None
        elif label == 'src':
            self.state = label
        elif label == 'ref':
            self.state = attr['translator']

    def endElement(self, label):
        global sentencepair
        if self.currentDoc is None:
            return
        if label == 'seg':
            if self.state == 'src':
                sentencepair.source.append(self.content)
            else:
                sentencepair.translate[self.state].append(self.content)

    def characters(self, content):
        self.content = content


parser = xml.sax.make_parser()  # 创建一个解析器的XMLreader对象
parser.setFeature(xml.sax.handler.feature_namespaces, 0)  # 从xml文件解析数据，关闭从命名空间解析数据
Handler = GetStorehouse()
parser.setContentHandler(Handler)
parser.parse("/Users/dream/Downloads/wmttest2022.zh-en.xml")
cut = pkuseg.pkuseg()

with open('dev.src', 'w') as src_o, open('dev.mt', 'w') as mt_o:
    for l1, l2 in zip(sentencepair.source, sentencepair.translate['B']):
        src_o.write(" ".join(cut.cut(l1)) + '\n')
        mt_o.write(l2 + '\n')
print(len(sentencepair.source), len(sentencepair.translate['B']))
