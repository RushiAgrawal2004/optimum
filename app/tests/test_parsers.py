from speed import parse_bench_json
from quality import parse_kl_output

BENCH_JSON = """
[
  {"n_prompt": 32, "n_gen": 0, "avg_ts": 284.83},
  {"n_prompt": 0, "n_gen": 16, "avg_ts": 81.19}
]
"""

def test_bench_parser_reads_gen_and_prompt_tps():
    result = parse_bench_json(BENCH_JSON)
    assert result["gen_tps"] == 81.19
    assert result["prompt_tps"] == 284.83

def test_bench_parser_survives_junk():
    assert parse_bench_json("") is None
    assert parse_bench_json("error: out of memory") is None

def test_bench_parser_no_gen_row_is_failure():
    assert parse_bench_json('[{"n_prompt": 32, "n_gen": 0, "avg_ts": 284.83}]') is None


KL_TEXT = """
Final estimate: PPL = 9.1234 +/- 0.05
KL divergence stats:
Mean KLD:  0.012345
99.9%   KLD:  0.087654
Same top p: 96.42%
"""

def test_kl_parser_reads_mean_and_agreement():
    q = parse_kl_output(KL_TEXT)
    assert q is not None
    assert abs(q.kl_mean - 0.012345) < 1e-9
    assert abs(q.top1_agree - 0.9642) < 1e-6

def test_kl_parser_survives_junk():
    assert parse_kl_output("") is None
    assert parse_kl_output("error: out of memory") is None
