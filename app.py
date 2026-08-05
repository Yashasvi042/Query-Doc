from flask import Flask, render_template, request, redirect, url_for
import os
from pypdf import PdfReader

app = Flask(__name__)

pdf_text = ""

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global pdf_text

    pdf = request.files["pdf"]

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], pdf.filename)
    pdf.save(filepath)

    print("PDF Saved At:", filepath)

    reader = PdfReader(filepath)

    pdf_text = ""

    print("Number of pages:", len(reader.pages))

    for i, page in enumerate(reader.pages):
        extracted = page.extract_text()

        print(f"\n------ Page {i+1} ------")
        print(extracted)

        if extracted:
            pdf_text += extracted

    print("\n========== FINAL PDF TEXT ==========")
    print(pdf_text)
    print("Length of pdf_text:", len(pdf_text))
    print("===================================")

    return redirect(url_for("home"))


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]

    print("Question:", question)
    print("Length of pdf_text:", len(pdf_text))

    if question.lower() in pdf_text.lower():
        return "Yes, I found something related to your question in the PDF."

    return "I couldn't find anything related to your question."
    
if __name__ == "__main__":
    app.run(debug=True)