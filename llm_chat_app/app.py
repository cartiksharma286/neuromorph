import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")

google_key = os.getenv("GOOGLE_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
provider = "demo"
model_name = None

if openai_key:
    from langchain_openai import ChatOpenAI

    model_name = "gpt-5.6"
    provider = "openai"
    llm = ChatOpenAI(
        model=model_name,
        api_key=openai_key,
    )
elif google_key:
    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = "gemini-3.1-pro-preview"
    provider = "google"
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.7,
        google_api_key=google_key,
    )
elif anthropic_key:
    from langchain_anthropic import ChatAnthropic

    model_name = "claude-3-5-sonnet-20240620"
    provider = "anthropic"
    llm = ChatAnthropic(
        model=model_name,
        temperature=0.7,
        anthropic_api_key=anthropic_key,
    )
else:
    llm = None

if llm:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful, thoughtful assistant. Answer clearly, briefly, and with a friendly tone. When relevant, provide actionable steps or examples."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
else:
    chain = None


def fallback_answer(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return "Please ask me a question and I will help answer it."
    return (
        "Demo mode is active because no provider API key was found. "
        f"For GPT-5.6 answers, add OPENAI_API_KEY to the .env file, then restart the app.\n\n"
        f"Your question: '{q}'\n\n"
        "I can still help by suggesting a concise response framework: define the problem, identify constraints, propose a few options, and recommend the best next step."
    )


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "mode": "live" if chain else "demo",
        "provider": provider,
        "model": model_name,
    })


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()

    if not question:
        return jsonify({"answer": "Please enter a question before sending."}), 400

    try:
        if chain is None:
            answer = fallback_answer(question)
        else:
            answer = chain.invoke({"question": question})
        return jsonify({"answer": answer})
    except Exception as exc:
        answer = fallback_answer(question) + f"\n\nSystem note: {exc}"
        return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5007")), debug=True)
