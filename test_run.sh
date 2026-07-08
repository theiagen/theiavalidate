uv run --project . theiavalidate validate \
    tests/phb/theiaprok/theiaprok_illumina_pe_v4-2-0.tsv \
    tests/phb/theiaprok/theiaprok_illumina_pe_v4-1-0.tsv \
    --config validation-criteria-v4-2-0-theiaprok-illumina-pe.yaml \
    --outdir out --prefix theiaprok_pe_4-2-0_v4-1-0
