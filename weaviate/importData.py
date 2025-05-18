import pandas as pd
import weaviate
from sentence_transformers import SentenceTransformer

# 1. Đọc dữ liệu
df = pd.read_csv('UpdatedResumeDataSet.csv')  # Cột: Category, Resume

# 2. Kết nối Weaviate
client = weaviate.Client("http://localhost:8080")

# 3. Tạo schema
# Xoá class cũ nếu cần
if client.schema.contains({"class": "Resume"}):
    client.schema.delete_class("Resume")

schema = {
    "classes": [{
        "class": "Resume",
        "description": "CV dataset",
        "vectorizer": "none",  # vì bạn sẽ tự vector hóa
        "properties": [
            {
                "name": "category",
                "dataType": ["text"]
            },
            {
                "name": "resume_text",
                "dataType": ["text"]
            }
        ]
    }]
}

client.schema.create(schema)

# 4. Vector hóa với SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# 5. Thêm dữ liệu vào Weaviate
for idx, row in df.iterrows():
    text = row["Resume"]
    vec = model.encode(text).tolist()
    obj = {
        "category": row["Category"],
        "resume_text": text
    }
    client.data_object.create(data_object=obj, class_name="Resume", vector=vec)

print("Import complete.")

