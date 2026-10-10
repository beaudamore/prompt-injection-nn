from prompt_injection_nn.tokenizer import Tokenizer

PATTERN = r"[A-Za-z]+|\d+|[^\sA-Za-z\d]+"


def make(max_tokens=512):
    vocab = {"<pad>": 0, "<unk>": 1, "Ignore": 2, "ignore": 3, "<|": 4, "im": 5, "_": 6, "start": 7, "|>": 8, "###": 9}
    return Tokenizer(PATTERN, vocab, max_tokens, 0, 1)


def test_special_tokens_are_split_like_the_notebook():
    assert make().tokenize("<|im_start|>system") == ["<|", "im", "_", "start", "|>", "system"]
    assert make().tokenize("### SYSTEM\n---") == ["###", "SYSTEM", "---"]


def test_case_is_preserved_and_unknowns_map_to_unk():
    tok = make()
    assert tok.encode("Ignore ignore IGNORE") == [2, 3, 1]


def test_encode_is_deterministic_and_truncates():
    tok = make(max_tokens=3)
    text = "Ignore ignore <|im_start|>"
    assert tok.encode(text) == tok.encode(text)
    assert len(tok.encode(text)) == 3


def test_empty_text_encodes_to_unk():
    assert make().encode("") == [1]


def test_windows_cover_the_tail_of_a_long_text():
    tok = make(max_tokens=4)
    text = " ".join(["ignore"] * 10) + " ###"
    windows = tok.windows(text, stride=2)
    assert all(len(w) == 4 for w in windows)
    assert windows[-1][-1] == 9, "the last window must end on the last token"
    assert tok.windows("ignore", stride=2) == [[3]]


def test_pad_shapes_and_values():
    tok = make()
    padded = tok.pad([[2, 3], [4]])
    assert padded.shape == (2, 2)
    assert padded[1].tolist() == [4, 0]
