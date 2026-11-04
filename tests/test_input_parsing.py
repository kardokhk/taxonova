"""Query parsing contract (parser).

The reader must return one record per input record, in order, with the
sequence as uppercase IUPAC letters (U kept: the k-mer step reads it as T),
alignment gaps and whitespace removed. It must read FASTA and FASTQ, plain or
gzip (detected from the bytes, not the file name), and must never drop a
record or return nothing from a non-empty file without saying why.
"""
import gzip
import logging
import string

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from taxonova.data.fasta import read_fasta_file

SEQ = "ACGTACGTTGCA"


def _read(tmp_path, text, name="q.fasta", mode="w", **kw):
    p = tmp_path / name
    if mode == "wb":
        p.write_bytes(text)
    else:
        p.write_text(text, encoding=kw.pop("encoding", "utf-8"), newline="")
    return read_fasta_file(str(p), expect_taxonomy=False)


def _ids_seqs(recs):
    return [(r[0], r[1]) for r in recs]


def test_multiline_crlf_blank_lines(tmp_path):
    text = ">a desc\r\nACGT\r\nacgt\r\n\r\n>b\r\nTTTT\r\n"
    assert _ids_seqs(_read(tmp_path, text)) == [("a", "ACGTACGT"), ("b", "TTTT")]


def test_lowercase_is_uppercased(tmp_path):
    assert _ids_seqs(_read(tmp_path, ">a\nacgtn\n")) == [("a", "ACGTN")]


def test_alignment_gaps_are_removed(tmp_path):
    # An aligned (gapped) query must give the same features as the ungapped read.
    assert _ids_seqs(_read(tmp_path, ">a\nAC--GT..AC-GT\n")) == [("a", "ACGTACGT")]


def test_internal_whitespace_is_removed(tmp_path):
    assert _ids_seqs(_read(tmp_path, ">a\nACGT ACGT\tAC\n")) == [("a", "ACGTACGTAC")]


def test_iupac_and_u_are_kept(tmp_path):
    assert _ids_seqs(_read(tmp_path, ">a\nACGURYSWKMBDHVN\n")) == [("a", "ACGURYSWKMBDHVN")]


def test_empty_record_is_kept_not_dropped(tmp_path):
    # Every input record must appear exactly once in the output.
    recs = _read(tmp_path, ">a\nACGT\n>empty\n>c\nGGGG\n>last_empty\n")
    assert [r[0] for r in recs] == ["a", "empty", "c", "last_empty"]
    assert [r[1] for r in recs] == ["ACGT", "", "GGGG", ""]


def test_header_without_id_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="line 3"):
        _read(tmp_path, ">a\nACGT\n>\nGGGG\n")


def test_header_with_space_after_marker_uses_first_word(tmp_path):
    assert [r[0] for r in _read(tmp_path, ">  idx some description\nACGT\n")] == ["idx"]


def test_duplicate_ids_are_kept_and_warned(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        recs = _read(tmp_path, ">a\nACGT\n>a\nGGGG\n")
    assert _ids_seqs(recs) == [("a", "ACGT"), ("a", "GGGG")]
    assert "duplicate" in caplog.text.lower()


def test_non_ascii_utf8_header_round_trips(tmp_path):
    assert [r[0] for r in _read(tmp_path, ">Ölstrøm_µ1 x\nACGT\n")] == ["Ölstrøm_µ1"]


def test_latin1_header_does_not_crash(tmp_path):
    recs = _read(tmp_path, ">caf\xe9 x\nACGT\n".encode("latin-1"), mode="wb")
    assert len(recs) == 1 and recs[0][1] == "ACGT"


def test_non_ascii_in_sequence_is_not_silently_deleted(tmp_path):
    # Deleting the character would join its flanks into k-mers that are not in the read.
    recs = _read(tmp_path, ">a\nACGTÄACGT\n")
    assert len(recs[0][1]) == 9


def test_gzip_detected_by_content_not_name(tmp_path):
    data = gzip.compress(b">a\nACGT\n")
    assert _ids_seqs(_read(tmp_path, data, name="q.fasta", mode="wb")) == [("a", "ACGT")]
    assert _ids_seqs(_read(tmp_path, data, name="q.fa.gz", mode="wb")) == [("a", "ACGT")]


def test_fastq_is_read(tmp_path):
    text = "@r1 desc\nACGT\n+\nIIII\n@r2\nggcc\n+r2\n@@@@\n"
    assert _ids_seqs(_read(tmp_path, text, name="q.fastq")) == [("r1", "ACGT"), ("r2", "GGCC")]


def test_fastq_gz_is_read(tmp_path):
    data = gzip.compress(b"@r1\nACGT\n+\nIIII\n")
    assert _ids_seqs(_read(tmp_path, data, name="q.fq.gz", mode="wb")) == [("r1", "ACGT")]


def test_truncated_fastq_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        _read(tmp_path, "@r1\nACGT\n+\n", name="q.fastq")


def test_non_fasta_text_is_an_error_not_zero_records(tmp_path):
    with pytest.raises(ValueError):
        _read(tmp_path, "sequence_id\tdomain\nx\tBacteria\n", name="q.tsv")


def test_empty_file_gives_zero_records(tmp_path):
    assert _read(tmp_path, "") == []


# --- property-based ------------------------------------------------------

ids = st.text(alphabet=string.ascii_letters + string.digits + "_.:|-", min_size=1, max_size=20)
seqs = st.text(alphabet="ACGTUNRYacgtun", min_size=0, max_size=200)
records = st.lists(st.tuples(ids, seqs), min_size=0, max_size=15)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(records=records, width=st.integers(1, 80), crlf=st.booleans(), gz=st.booleans(),
       blank=st.booleans(), gaps=st.booleans())
def test_fasta_round_trip(tmp_path, records, width, crlf, gz, blank, gaps):
    nl = "\r\n" if crlf else "\n"
    out = []
    for sid, s in records:
        out.append(f">{sid} some description")
        g = "-".join(s) if gaps and s else s
        out.extend(g[i:i + width] for i in range(0, len(g), width))
        if blank:
            out.append("")
    text = nl.join(out) + (nl if out else "")
    data = gzip.compress(text.encode()) if gz else text.encode()
    p = tmp_path / "rt.fasta"
    p.write_bytes(data)
    got = read_fasta_file(str(p), expect_taxonomy=False)
    assert [(r[0], r[1]) for r in got] == [(sid, s.upper()) for sid, s in records]


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(records=st.lists(st.tuples(ids, st.text(alphabet="ACGTN", min_size=1, max_size=150)), max_size=10))
def test_fastq_round_trip(tmp_path, records):
    text = "".join(f"@{sid}\n{s}\n+\n{'I' * len(s)}\n" for sid, s in records)
    p = tmp_path / "rt.fastq"
    p.write_text(text)
    got = read_fasta_file(str(p), expect_taxonomy=False)
    assert [(r[0], r[1]) for r in got] == records


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(blob=st.binary(max_size=400))
def test_arbitrary_bytes_never_crash_unexpectedly(tmp_path, blob):
    """Garbage either parses or raises ValueError; never another exception."""
    p = tmp_path / "junk.fasta"
    p.write_bytes(blob)
    try:
        read_fasta_file(str(p), expect_taxonomy=False)
    except ValueError:
        pass
