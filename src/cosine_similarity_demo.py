from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import clean_text

# -----------------------------
# Load Sentence Transformer
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Example Sentences
# -----------------------------
sentence1 = "play the music"

sentence2 = "start playing songs"

sentence3 = "increase the volume"

sentence4 = "what is today's weather"

# -----------------------------
# Clean Sentences
# -----------------------------
sentence1 = clean_text(sentence1)
sentence2 = clean_text(sentence2)
sentence3 = clean_text(sentence3)
sentence4 = clean_text(sentence4)

# -----------------------------
# Generate Embeddings
# -----------------------------
embedding1 = model.encode([sentence1])

embedding2 = model.encode([sentence2])

embedding3 = model.encode([sentence3])

embedding4 = model.encode([sentence4])

# -----------------------------
# Calculate Similarities
# -----------------------------
similarity_1_2 = cosine_similarity(
    embedding1,
    embedding2
)[0][0]

similarity_1_3 = cosine_similarity(
    embedding1,
    embedding3
)[0][0]

similarity_1_4 = cosine_similarity(
    embedding1,
    embedding4
)[0][0]

# -----------------------------
# Display Results
# -----------------------------
print("=" * 60)
print("Cosine Similarity Results")
print("=" * 60)

print(f"\nSentence 1 : {sentence1}")
print(f"Sentence 2 : {sentence2}")
print(f"Similarity : {similarity_1_2:.4f}")

print("\n" + "-" * 60)

print(f"\nSentence 1 : {sentence1}")
print(f"Sentence 3 : {sentence3}")
print(f"Similarity : {similarity_1_3:.4f}")

print("\n" + "-" * 60)

print(f"\nSentence 1 : {sentence1}")
print(f"Sentence 4 : {sentence4}")
print(f"Similarity : {similarity_1_4:.4f}")