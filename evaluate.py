"""Run the small manual evaluation suite after indexing a document set."""
import json
from pathlib import Path

from src.answer import answer_question
from src.retrieve import retrieve

EVALUATION_PATH = Path("tests/evaluation_questions.json")


def main() -> None:
    cases = json.loads(EVALUATION_PATH.read_text())
    passed = 0
    for case in cases:
        chunks = retrieve(case["question"])
        answer = answer_question(case["question"])
        answer_ok = answer.grounded == case["should_answer"]
        pages_ok = True
        if case["should_answer"] and case.get("expected_pages"):
            pages_ok = bool(set(case["expected_pages"]) & {chunk.page for chunk in chunks})
        status = "PASS" if answer_ok and pages_ok else "FAIL"
        passed += status == "PASS"
        print("{} | {}".format(status, case["question"]))
    print("{} / {} evaluation cases passed".format(passed, len(cases)))


if __name__ == "__main__":
    main()
