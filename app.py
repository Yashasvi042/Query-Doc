from flask import Flask, render_template, request, redirect, url_for
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

app = Flask(__name__)

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Variables to store PDF data
pdf_text = ""
chunks = []
faiss_index = None

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global pdf_text, chunks, faiss_index

    # Get uploaded PDF
    pdf = request.files["pdf"]

    # Save PDF
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], pdf.filename)
    pdf.save(filepath)

    # Read PDF
    reader = PdfReader(filepath)

    pdf_text = ""

    # Extract text from every page
    for i, page in enumerate(reader.pages):
        extracted = page.extract_text()

        print(f"\n------ Page {i + 1} ------")
        print(extracted)

        if extracted:
            pdf_text += extracted

    print("\n========== FINAL PDF TEXT ==========")
    print("Length of pdf_text:", len(pdf_text))

    # Split PDF text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(pdf_text)

    print("Number of chunks:", len(chunks))

    # Print chunks
    for i, chunk in enumerate(chunks):
        print(f"\n------ Chunk {i + 1} ------")
        print(chunk)

    # Convert chunks into embeddings
    embeddings = model.encode(chunks)

    print("\nEmbeddings shape:", embeddings.shape)

    # Create FAISS index
    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatL2(dimension)

    # Add embeddings to FAISS
    faiss_index.add(embeddings)

    print("Number of vectors in FAISS:", faiss_index.ntotal)

    return redirect(url_for("home"))


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]

    print("\nQuestion:", question)

    if faiss_index is None:
        return "Please upload a PDF first."

    # Convert the question into an embedding
    question_embedding = model.encode([question])

    # Search FAISS for the 3 most relevant chunks
    distances, indices = faiss_index.search(question_embedding, 3)

    print("\n========== RELEVANT CHUNKS ==========")

    relevant_chunks = []

    for i, index in enumerate(indices[0]):
        chunk = chunks[index]

        print(f"\n------ Result {i + 1} ------")
        print("Distance:", distances[0][i])
        print("Chunk:")
        print(chunk)

        relevant_chunks.append(chunk)

    return "<br><br>".join(relevant_chunks)
    
if __name__ == "__main__":
    app.run(debug=True)