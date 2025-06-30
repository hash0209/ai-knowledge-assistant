from sentence_transformers import SentenceTransformer

# Use a very small model
model = SentenceTransformer("all-MiniLM-L6-v2")  # this will download & load the model

# Try generating an embedding
embedding = model.encode("Hello world", device="cpu")

print("Embedding generated successfully. Length:", len(embedding))