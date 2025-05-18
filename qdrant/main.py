from qdrant_client import QdrantClient
from qdrant_client.http import models
from faker import Faker
from faker_food import FoodProvider
from sentence_transformers import SentenceTransformer
import numpy as np
from pprint import pprint
import uuid

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Collection name
my_collection = "food_collection"

# Delete collection if it exists
if client.collection_exists(my_collection):
    client.delete_collection(my_collection)

# Initialize sentence transformer model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')  # This model produces 384-dimensional vectors

# Create collection with vector size matching the embedding model
client.create_collection(
    collection_name=my_collection,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
)

print(f"✅ Collection '{my_collection}' has been created.")

# Initialize Faker for generating payloads
fake = Faker()
fake.add_provider(FoodProvider)

# Generate payloads
num_records = 1000
payloads = []
for _ in range(num_records):
    payload = {
        "restaurant": fake.name(),
        "ethnic_category": fake.ethnic_category(),
        "dish_description": fake.dish_description(),
        "dish": fake.dish(),
        "url": fake.url(),
        "year": fake.year(),
        "country": fake.country()
    }
    payloads.append(payload)

# Combine relevant fields for embedding (e.g., dish and description)
texts_to_embed = [f"{p['dish']} {p['dish_description']}" for p in payloads]

# Generate embeddings
embeddings = model.encode(texts_to_embed, show_progress_bar=True)

# Prepare data for upsert
ids = [str(uuid.uuid4()) for _ in range(num_records)]  # Unique IDs for each record
vectors = embeddings.tolist()

# Upsert vectors and payloads into Qdrant
upsert_result = client.upsert(
    collection_name=my_collection,
    points=models.Batch(
        ids=ids,
        vectors=vectors,
        payloads=payloads
    )
)

print(f"✅ Upserted {num_records} points with payloads.")

# Example query: Embed a sample text for search
query_text = "spicy Vietnamese pho"
query_vector = model.encode([query_text])[0].tolist()

# Perform a simple search
search_results = client.search(
    collection_name=my_collection,
    query_vector=query_vector,
    limit=3
)

print("\nSearch Results:")
for result in search_results:
    print(f"ID: {result.id}, Score: {result.score}, Payload: {result.payload}")

# Filter search by country (e.g., Australia)
australian_filter = models.Filter(
    must=[models.FieldCondition(key="country", match=models.MatchValue(value="Australia"))]
)

filtered_search = client.search(
    collection_name=my_collection,
    query_vector=query_vector,
    query_filter=australian_filter,
    limit=2
)

print("\nFiltered Search Results (Australia):")
for result in filtered_search:
    print(f"ID: {result.id}, Score: {result.score}, Payload: {result.payload}")

# Recommendation based on a positive example
recommendation = client.recommend(
    collection_name=my_collection,
    positive=[ids[0]],  # Use the first ID as a positive example
    limit=3
)

print("\nRecommendation Results:")
for result in recommendation:
    print(f"ID: {result.id}, Score: {result.score}, Payload: {result.payload}")

# Recommendation with positive and negative examples
p_and_n_recommendation = client.recommend(
    collection_name=my_collection,
    positive=[ids[0]],
    negative=[ids[1]],
    limit=3
)

print("\nPositive and Negative Recommendation Results:")
for result in p_and_n_recommendation:
    print(f"ID: {result.id}, Score: {result.score}, Payload: {result.payload}")

# Recommendation with a score threshold
fine_tune_recommendation = client.recommend(
    collection_name=my_collection,
    positive=[ids[2]],
    negative=[ids[1], ids[3]],
    score_threshold=0.22,
    limit=3
)

print("\nFine-Tuned Recommendation Results (Score Threshold):")
for result in fine_tune_recommendation:
    print(f"ID: {result.id}, Score: {result.score}, Payload: {result.payload}")

# Recommendation with a country filter (e.g., Vietnam)
vietnam_filter = models.Filter(
    must=[models.FieldCondition(key="country", match=models.MatchValue(value="Vietnam"))]
)

filter_recommendation = client.recommend(
    collection_name=my_collection,
    query_filter=vietnam_filter,
    positive=[ids[0]],
    negative=[ids[1]],
    limit=5
)

print("\nFiltered Recommendation Results (Vietnam):")
for result in filter_recommendation:
    print(f"ID: {result.id}, Score: {result.score}, Payload: {result.payload}")