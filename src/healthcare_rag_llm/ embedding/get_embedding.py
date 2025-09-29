from FlagEmbedding import BGEM3FlagModel

def get_embedding(text:str):
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    return model.encode(text)
    

if __name__ == "__main__":
    text = "Hello, world!"
    embedding = get_embedding(text)
    print(embedding)