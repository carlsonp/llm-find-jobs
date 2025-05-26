from sentence_transformers import SentenceTransformer

# https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
# downloads model file
# to /root/.cache/huggingface/hub/
model = SentenceTransformer("all-MiniLM-L12-v2")

model.encode("test")
