"""Command-line entry point for the RAG demo."""
import argparse
import json

from src.answer import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question of the indexed PDFs.")
    parser.add_argument("question", help="Question to answer from the document set")
    args = parser.parse_args()
    result = answer_question(args.question)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
