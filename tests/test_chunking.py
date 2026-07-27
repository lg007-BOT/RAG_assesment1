from src.chunking import PageText, chunks_from_pages


def test_chunks_keep_page_metadata_and_overlap():
    sentence = "This is a test sentence with several words for chunking."
    pages = [PageText(document="sample.pdf", page=1, text=" ".join([sentence] * 200))]
    chunks = chunks_from_pages(pages)
    assert len(chunks) > 1
    assert chunks[0]["document"] == "sample.pdf"
    assert chunks[0]["page"] == 1
    assert chunks[0]["text"].split()[-10:] == chunks[1]["text"].split()[:10]
