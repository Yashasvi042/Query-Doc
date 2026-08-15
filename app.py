from flask import Flask, render_template, request, redirect, url_for
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import ollama

app = Flask(__name__)


model = SentenceTransformer("all-MiniLM-L6-v2")


pdf_text = ""
chunks = []
faiss_index = None


UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global pdf_text, chunks, faiss_index

    pdf = request.files.get("pdf")

    
    if not pdf or pdf.filename == "":
        return "Please select a PDF file."

   
    if not pdf.filename.lower().endswith(".pdf"):
        return "Please upload a PDF file."

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        pdf.filename
    )

    pdf.save(filepath)

    
    reader = PdfReader(filepath)

    pdf_text = ""

    
    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            pdf_text += extracted

    if not pdf_text.strip():
        return "Could not extract text from this PDF."

    print("\n========== PDF UPLOADED ==========")
    print("PDF text length:", len(pdf_text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(pdf_text)

    print("Number of chunks:", len(chunks))

    
    embeddings = model.encode(chunks)

    print("Embeddings shape:", embeddings.shape)

   
    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatL2(dimension)

   
    faiss_index.add(embeddings)

    print("Vectors stored in FAISS:", faiss_index.ntotal)
    print("Document ready for questions.")


    return redirect(url_for("home"))


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()

    if not question:
        return "Please enter a question."

    #
    if faiss_index is None:
        return "Please upload a PDF first."

    print("\n========== QUESTION ==========")
    print(question)

   
    question_embedding = model.encode([question])

    
    distances, indices = faiss_index.search(
        question_embedding,
        8
    )

  
    relevant_chunks = []

    for index in indices[0]:
        start = max(0, index - 1)
        end = min(len(chunks), index + 2)

        for i in range(start, end):
            if chunks[i] not in relevant_chunks:
                relevant_chunks.append(chunks[i])
   
    context = "\n\n".join(relevant_chunks)

    print("\n========== RELEVANT CONTEXT RETRIEVED ==========")

    for i, chunk in enumerate(relevant_chunks):
        print(f"\n--- Relevant Chunk {i + 1} ---")
        print(chunk)

  
    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the PDF context below.

If the answer is not present in the context, say:

"I could not find the answer in the uploaded PDF."

Do not make up information.

PDF CONTEXT:
{context}

USER QUESTION:
{question}

Answer clearly and concisely.
"""

    print("\n========== SENDING TO LLAMA ==========")

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    
    answer = response["message"]["content"]

    print("\n========== LLAMA ANSWER ==========")
    print(answer)

    return answer


if __name__ == "__main__":
    app.run(debug=True)