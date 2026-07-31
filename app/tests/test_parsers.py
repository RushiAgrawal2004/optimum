from pathlib import Path
from speed import parse_bench_json
from quality import parse_kl_output

FIXTURES = Path(__file__).parent / "fixtures"


def test_bench_parser_reads_real_output():
    raw = (FIXTURES / "bench_output.json").read_text()
    result = parse_bench_json(raw)
    assert result is not None
    assert 0 < result["gen_tps"] < 10000
    assert 0 < result["prompt_tps"] < 10000

def test_bench_parser_survives_junk():
    assert parse_bench_json("") is None
    assert parse_bench_json("error: out of memory") is None

def test_bench_parser_no_gen_row_is_failure():
    assert parse_bench_json('[{"n_prompt": 32, "n_gen": 0, "avg_ts": 284.83}]') is None


def test_kl_parser_reads_real_output():
    raw = (FIXTURES / "kl_output.txt").read_text()
    q = parse_kl_output(raw)
    assert q is not None
    assert 0 < q.kl_mean < 1
    assert 0 < q.top1_agree < 1

def test_kl_parser_survives_junk():
    assert parse_kl_output("") is None
    assert parse_kl_output("error: out of memory") is None
