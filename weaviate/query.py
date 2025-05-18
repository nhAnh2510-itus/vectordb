from sentence_transformers import SentenceTransformer
import weaviate

# Kết nối lại Weaviate
client = weaviate.Client('http://localhost:8080')
model = SentenceTransformer('all-MiniLM-L6-v2')

# Gợi ý mô tả JD để test semantic matching
job_descriptions = [
    'Seeking a backend engineer familiar with cloud-native architectures, containerization, and CI/CD practices and have knowlegde with bigdata tool.'
]

for jd in job_descriptions:
    q_vec = model.encode(jd).tolist()
    res = (
        client.query.get('Resume', ['category', 'resume_text'])
              .with_near_vector({'vector': q_vec})
              .with_limit(10)
              .do()
    )
    print(f"\n=== JD: {jd}\n--- Top Matches: ---")
    for item in res['data']['Get']['Resume']:
        print('-', item['category'], '|', item['resume_text'])
