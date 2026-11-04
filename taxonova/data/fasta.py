import itertools
import re
import zlib
from collections import Counter
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

def parse_sequence_id(header: str) -> str:
    """Extract sequence ID from FASTA header, handling various formats."""
    match = re.match(r'^[>@]?(\S+)', header)
    if match:
        return match.group(1)
    return header.lstrip('>@').strip()

def parse_taxonomy(header: str, expect_taxonomy: bool = True) -> Dict[str, str]:
    """Parse taxonomy information from FASTA header.
    
    Expected format:
    >sequence_id d__Bacteria; p__Phylum; c__Class; o__Order; f__Family; g__Genus; s__Species name
    """
    taxonomy = {
        "domain": "unknown",
        "phylum": "unknown",
        "class": "unknown",
        "order": "unknown",
        "family": "unknown",
        "genus": "unknown",
        "species": "unknown"
    }
    
    try:
        full_header = header.lstrip('>')
        parts = full_header.split()
        if len(parts) < 2:
            if expect_taxonomy:
                logger.error(f"No taxonomy found in header: {header}")
            return taxonomy

        taxonomy_string = ' '.join(parts[1:])
        taxonomy_parts = [p.strip() for p in taxonomy_string.split(';')]
        
        first_part = taxonomy_parts[0]
        if 'd__' in first_part:
            domain = first_part.split('d__')[1].strip()
            taxonomy["domain"] = domain
        
        for part in taxonomy_parts:
            part = part.strip()
            if '__' not in part:
                continue
                
            prefix, value = part.split('__', 1)
            prefix = prefix.strip().lower()
            value = value.strip()
            
            if prefix == 'd':
                taxonomy["domain"] = value
            elif prefix == 'p':
                taxonomy["phylum"] = value
            elif prefix == 'c':
                taxonomy["class"] = value
            elif prefix == 'o':
                taxonomy["order"] = value
            elif prefix == 'f':
                taxonomy["family"] = value
            elif prefix == 'g':
                taxonomy["genus"] = value
            elif prefix == 's':
                taxonomy["species"] = value
                
    except Exception as e:
        logger.warning(f"Error parsing taxonomy from header: {header}. Error: {str(e)}")
        
    return taxonomy

GZIP_MAGIC = b"\x1f\x8b"
#: Removed from sequence lines: alignment gaps and any whitespace.
_STRIP = re.compile(r"[\s.\-]+")
#: Characters other than these are kept but replaced by N, so a stray symbol
#: breaks k-mers (like N) instead of being deleted and joining its flanks.
_ALLOWED = re.compile(r"[^ACGTURYSWKMBDHVN]")


def _open_text(file_path):
    """Open a FASTA/FASTQ file as text. gzip is detected from the first bytes,
    not the file name. Bytes that are not UTF-8 (a latin-1 header, say) are
    carried through with surrogateescape, so an id is written back exactly."""
    with open(file_path, "rb") as fh:
        magic = fh.read(2)
    if magic == GZIP_MAGIC:
        import gzip
        return gzip.open(file_path, "rt", encoding="utf-8", errors="surrogateescape", newline=None)
    return open(file_path, "r", encoding="utf-8", errors="surrogateescape", newline=None)


def clean_sequence(raw: str) -> str:
    """Uppercase IUPAC letters with gaps and whitespace removed."""
    return _ALLOWED.sub("N", _STRIP.sub("", raw).upper())


def _record_id(header: str, line_no: int, path) -> str:
    seq_id = parse_sequence_id(header.strip())
    if not seq_id or seq_id in (">", "@"):
        raise ValueError(f"{path}: record at line {line_no} has no sequence id")
    return seq_id


def _iter_fasta(lines, path):
    seq_id = header = None
    chunks = []
    for line_no, line in lines:
        if line.startswith(">"):
            if seq_id is not None:
                yield seq_id, header, clean_sequence("".join(chunks))
            header = line.rstrip("\r\n")
            seq_id = _record_id(header[1:].lstrip(), line_no, path)
            chunks = []
        elif not line.strip():
            continue
        elif line.startswith(";"):
            continue  # old-style FASTA comment line
        elif seq_id is None:
            raise ValueError(f"{path}: line {line_no} is sequence data before any '>' header; "
                             "the file is not FASTA or FASTQ")
        else:
            chunks.append(line)
    if seq_id is not None:
        yield seq_id, header, clean_sequence("".join(chunks))


def _iter_fastq(lines, path):
    lines = iter(lines)
    for line_no, head in lines:
        if not head.strip():
            continue
        if not head.startswith("@"):
            raise ValueError(f"{path}: line {line_no}: expected a FASTQ '@' header")
        seq_id = _record_id(head[1:].lstrip(), line_no, path)
        try:
            _, seq = next(lines)
            plus_no, plus = next(lines)
            _, qual = next(lines)
        except StopIteration:
            raise ValueError(f"{path}: FASTQ record {seq_id!r} at line {line_no} is truncated") from None
        if not plus.startswith("+"):
            raise ValueError(f"{path}: line {plus_no}: expected '+' in FASTQ record {seq_id!r} "
                             "(multi-line FASTQ is not supported)")
        seq, qual = seq.strip(), qual.strip()
        if len(seq) != len(qual):
            raise ValueError(f"{path}: FASTQ record {seq_id!r}: sequence and quality lengths differ")
        yield seq_id, head.rstrip("\r\n"), clean_sequence(seq)


def read_fasta_file(file_path: str, expect_taxonomy: bool = True) -> List[Tuple[str, str, Dict[str, str]]]:
    """Read FASTA or FASTQ (plain or gzip) into (id, sequence, taxonomy) tuples.

    Sequences are uppercased, and alignment gaps ('-', '.') and whitespace are
    removed. For queries (``expect_taxonomy=False``) every record is returned,
    in input order, including empty ones, so that each appears once in the
    output. Reference reading (``expect_taxonomy=True``) skips empty records
    with a warning. Duplicate ids are kept and reported. Any I/O or format
    problem raises ValueError naming the file and line.
    """
    try:
        with _open_text(file_path) as fh:
            numbered = enumerate(fh, 1)
            first = None
            buffered = []
            for line_no, line in numbered:
                buffered.append((line_no, line))
                if line.strip():
                    first = line.lstrip()[0]
                    break
            if first is None:
                return []
            lines = itertools.chain(buffered, numbered)
            if first == "@":
                records = list(_iter_fastq(lines, file_path))
            elif first in ">;":
                records = list(_iter_fasta(lines, file_path))
            else:
                raise ValueError(f"{file_path}: not FASTA or FASTQ (first character {first!r})")
    except (OSError, EOFError, UnicodeError, zlib.error) as e:
        if isinstance(e, FileNotFoundError):
            raise
        raise ValueError(f"Error reading {file_path}: {e}") from e
    except ValueError as e:
        logger.error(str(e))
        raise

    sequences = []
    for seq_id, header, seq in records:
        if expect_taxonomy and not seq:
            logger.warning(f"Skipping reference record {seq_id!r}: empty sequence")
            continue
        sequences.append((seq_id, seq, parse_taxonomy(header, expect_taxonomy)))

    counts = Counter(r[0] for r in sequences)
    dups = [i for i, n in counts.items() if n > 1]
    if dups:
        logger.warning(f"{file_path}: {len(dups)} duplicate sequence id(s), e.g. {dups[0]!r}; "
                       "each record is kept and classified, so output rows share ids")

    if sequences and expect_taxonomy and sequences[0][2]["domain"] == "unknown":
        logger.warning(f"First sequence domain parsed as unknown. Original header: {sequences[0][0]}")
    return sequences
