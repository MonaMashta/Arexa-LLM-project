from flask import Flask, request, jsonify, render_template


from dotenv import load_dotenv
load_dotenv()

from rag_system import run_pipeline

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query")
def query():
    user_q = request.args.get("q")

    results = run_pipeline(user_q)

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)