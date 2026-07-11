uv run --project . theiavalidate validate \
    tests/phb/theiaprok/pe/theiaprok_illumina_pe_v4-2-0.tsv \
    tests/phb/theiaprok/pe/theiaprok_illumina_pe_v4-1-0.tsv \
    --preset theiaprok_pe \
    --key1 entity:theiaprok_illumina_pe_v4-2-0_id \
    --key2 entity:theiaprok_illumina_pe_v4-1-0_id \
    --outdir out --prefix theiaprok_pe_4-2-0_v4-1-0


# To run experimental qwen3 demo
# first,  you need a hugging face account and token
# export HF_TOKEN=xxxx
# PYTHONPATH=src uv run --no-project python -m theiavalidate.age