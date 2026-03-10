from flask import Flask, request, jsonify, render_template
from vision_pipeline import run_pipeline
from complexity_extractor import ComplexityAnalyzer

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/run_pipeline", methods=["POST"])
def run_pipeline_api():
    data = request.json
    b = int(data.get("b", 2))
    h = int(data.get("h", 64))
    w = int(data.get("w", 64))
    c = int(data.get("c", 3))
    L = int(data.get("L", 5))
    d = int(data.get("d", 3))
    p = int(data.get("p", 10))

    results = run_pipeline(b, h, w, c, L, d, p)
    return jsonify(results)


@app.route("/run_complexity", methods=["POST"])
def run_complexity_api():
    data = request.json

    analyzer = ComplexityAnalyzer("vision_pipeline.py")
    report = analyzer.analyze()  # could be dict or string

    # Wrap in "complexity" key
    if isinstance(report, dict):
        complexity_value = report
    else:
        complexity_value = {"message": str(report)}

    return jsonify({
        "complexity": complexity_value,   # <-- required by JS
        "parameters": data
    })


if __name__ == "__main__":
    app.run(debug=True)