import importlib
import sys
import time
 
importlib.reload(sys)
time1 = time.time()
 
import os.path
from pdfminer.pdfparser import  PDFParser,PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal,LAParams
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed


def parse(pdf_path,text_path):
    fp = open(pdf_path,'rb')

    #Create a PDF parser
    parser = PDFParser(fp)

    #Create a document object
    doc = PDFDocument()

    #Connect the parser with the doc object
    parser.set_document(doc)
    doc.set_parser(parser)

    #Initialization
    doc.initialize()

    if not doc.is_extractable:
        raise PDFTextExtractionNotAllowed
    else:
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr,laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr,device)

        for page in doc.get_pages():
            interpreter.process_page(page)
            layout = device.get_result()

            for x in layout:
                if (isinstance(x,LTTextBoxHorizontal)):
                    with open(text_path,'a',encoding='utf-8', errors='strict') as f:
                        result = x.get_text()
                        #print(result)
                        f.write(result + "\n")
if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    pdf_path = os.path.join(
    base_dir,
    "..", "..",  # back to root
    "data", "Childrens Evolution of Care","State", "Medicaid Updates", "mu_no01_jan21_pr.pdf"
)
    print(pdf_path)
    txt_path = os.path.join(base_dir, "test.txt")
    parse(pdf_path=pdf_path,text_path=txt_path)
    #time2 = time.time()


