"""Label hygiene for species and genus outputs.

Pure functions over label strings and a reference taxonomy built from the training lineages. Nothing here touches a
model; the functions post-process whatever a classifier (the network, a nearest-neighbour step or a stacker) emits.

Rules:

* **Cross-genus group rule.** A species-level output is a single species, or a group ``[a|b|...]`` whose members all
  belong to one genus. When the members span genera, the species output is restricted to the members inside the
  genus call if that subset is itself a species taxon of the reference under that genus; otherwise the species rank
  abstains and the lineage stops at the genus call (the shared higher-rank taxon, which may itself be a reference
  genus group).
* **Rank-path consistency.** The species call must be a child of the genus call in the reference. When it is not,
  the species abstains. The genus is never rewritten from the species (the 26.10.1 consistency step did that and
  lowered genus F1).
* **Placeholders.** A species name in open nomenclature (``sp.``, ``uncultured``, ``metagenome``, ``bacterium``, ...)
  or an ambiguous name is not presented as a species when a named alternative exists: a group drops its placeholder
  members, and a placeholder single species is replaced by the first named, path-consistent alternative supplied by
  the caller. Without a named alternative the species abstains. Numbered GTDB-style species (``sp004555055``) are
  species in Greengenes2 and are not placeholders (see :func:`placeholder_class`).
* **Ambiguity runs in reference records.** vsearch scores N as a match, so a reference record with a long run of
  ambiguity codes can tie with any query. :func:`max_ambiguity_run` and :func:`ambiguous_records` find such records;
  nearest-neighbour steps exclude them (:func:`drop_hits_to`).
* **Genus decision.** :func:`knn_genus_vote` and :func:`agreement_genus` combine the network genus with the
  nearest-neighbour hits; see their docstrings.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

RANKS = ("domain", "phylum", "class", "order", "family", "genus", "species")
PFX = ("d__", "p__", "c__", "o__", "f__", "g__", "s__")
G, SP = 5, 6
NOCALL = {"", "unknown", "unassigned", "unclassified", "none", "na"}

# ------------------------------------------------------------------------------------------------ label strings


def is_nocall(label: Optional[str]) -> bool:
    return label is None or label.strip().lower() in NOCALL


def is_group(label: str) -> bool:
    """True for a group label ``[a|b|...]``: in brackets, with two or more non-empty members separated by ``|``.
    A bracketed single name such as the genus ``[Eubacterium]`` is a name, not a group."""
    if not label or len(label) <= 2 or label[0] != "[" or label[-1] != "]" or "|" not in label:
        return False
    return sum(1 for m in label[1:-1].split("|") if m.strip()) >= 2


def members(label: str) -> List[str]:
    """Members of a group label ``[a|b]``, or ``[label]`` for any other label; ``[]`` for no call."""
    if is_nocall(label):
        return []
    if is_group(label):
        return [m.strip() for m in label[1:-1].split("|") if m.strip()]
    return [label.strip()]


def group_label(names: Iterable[str]) -> str:
    """Canonical label of a member set: the single name, or ``[a|b]`` with members sorted."""
    s = sorted(set(names))
    if not s:
        return ""
    return s[0] if len(s) == 1 else "[" + "|".join(s) + "]"


def parse_lineage(lineage: str) -> List[str]:
    """``'d__X; p__Y; ...'`` (or tab/semicolon separated bare names) to 7 bare names, '' where empty."""
    v = [x.strip() for x in lineage.replace("\t", ";").split(";")]
    v = (v + [""] * 7)[:7]
    out = []
    for x in v:
        if x[:3] in PFX:
            x = x[3:]
        out.append("" if is_nocall(x) else x)
    return out


def format_lineage(labels: Sequence[str]) -> str:
    return "; ".join(p + (x or "") for p, x in zip(PFX, labels))


# ------------------------------------------------------------------------------------------------ placeholders

OPEN_WORDS = {"sp", "sp.", "bacterium", "uncultured", "unidentified", "metagenome", "environmental", "unknown",
              "unclassified", "cf.", "cf", "aff.", "aff", "str.", "n.", "group", "clone", "endosymbiont", "symbiont"}
OPEN_PREFIXES = ("uncultured", "unidentified", "metagenome", "endosymbiont")
_NAMED_RE = re.compile(r"[a-z][a-z-]*(_[A-Z]+)?")


def _epithet(species: str, genus: str) -> str:
    if genus and species.startswith(genus + " "):
        return species[len(genus) + 1:]
    return species.split(" ", 1)[1] if " " in species else ""


def placeholder_class(species: str, genus: str = "") -> str:
    """'' (empty), 'open', 'numbered', 'ambiguous' or 'named' for one species name (not a group).

    Same rule as the one used to prepare the training labels, with the genus taken from the name when not given.
    'open' and 'ambiguous' are placeholders."""
    if not species:
        return ""
    if not genus:
        w = species.split(" ")
        genus = " ".join(w[:2]) if w[0] == "Candidatus" and len(w) > 2 else w[0]
    ep = _epithet(species, genus)
    words = [w.lower() for w in ep.split()]
    if not ep or "(" in ep or any(w in OPEN_WORDS for w in words) or any(w.startswith(OPEN_PREFIXES) for w in words):
        return "open"
    if re.search(r"\d", ep):
        return "numbered"
    g = genus[len("Candidatus "):] if genus.startswith("Candidatus ") else genus
    g = g.replace("Candidatus_", "")
    if not _NAMED_RE.fullmatch(ep) or ep.lower() == g.lower():
        return "ambiguous"
    return "named"


def is_placeholder(species: str, genus: str = "") -> bool:
    return placeholder_class(species, genus) in ("open", "ambiguous")


# ------------------------------------------------------------------------------------------------ reference


class RefTaxonomy:
    """Reference taxonomy from training lineages (each a sequence of 7 labels, '' where empty).

    Keeps, as written in the reference: the valid (genus, species) label pairs, the upper path of each genus label,
    and, per species member name, the genus member names it belongs to."""

    def __init__(self, lineages: Iterable[Sequence[str]]):
        self.pairs: Set[Tuple[str, str]] = set()
        self.species_by_genus: Dict[str, Set[str]] = defaultdict(set)
        self.genus_paths: Dict[str, Counter] = defaultdict(Counter)
        self.member_genera: Dict[str, Set[str]] = defaultdict(set)
        self.genus_members: Set[str] = set()
        self.paths: Set[Tuple[str, ...]] = set()
        self._parents: Dict[Tuple[int, str], Set[Tuple[str, ...]]] = defaultdict(set)
        for lin in lineages:
            lab = [("" if is_nocall(x) else x) for x in list(lin)[:7]]
            g, s = lab[G], lab[SP]
            for r in range(1, 7):
                if lab[r]:
                    self._parents[(r, lab[r])].add(tuple(lab[:r]))
            if all(lab):
                self.paths.add(tuple(lab))
            if g:
                self.genus_paths[g][tuple(lab[:G])] += 1
                self.genus_members.update(members(g))
            if g and s:
                self.pairs.add((g, s))
                self.species_by_genus[g].add(s)
                gm = members(g)
                for m in members(s):
                    # genus of a species member: the genus member that prefixes its name, else (single genus) that
                    # genus, else every genus member of the record
                    pre = [x for x in gm if m.startswith(x + " ")]
                    if pre:
                        self.member_genera[m].update(pre)
                    elif len(gm) == 1 and genus_token_agrees(m, gm[0]):
                        # a record whose genus disagrees with the species name (a source-organism name, for
                        # example) does not define the genus of that species
                        self.member_genera[m].update(gm)
        # member sets of the species taxa under each genus label, for subset matching
        self._by_set: Dict[Tuple[str, frozenset], str] = {}
        for g, ss in self.species_by_genus.items():
            for s in ss:
                self._by_set[(g, frozenset(members(s)))] = s

    @classmethod
    def from_fasta(cls, path: str) -> "RefTaxonomy":
        """Training FASTA with headers ``>id<TAB or space>d__...; s__...``."""
        return cls(lineages_from_fasta(path).values())

    def multi_parent_names(self) -> Dict[str, List[str]]:
        """Per rank, the names that sit under more than one parent path (the same name under two paths is two
        taxa). Labels are keyed by full path in this class, so these names are counted, not merged."""
        out: Dict[str, List[str]] = defaultdict(list)
        for (r, name), ps in self._parents.items():
            if len(ps) > 1:
                out[RANKS[r]].append(name)
        return {k: sorted(v) for k, v in out.items()}

    def is_path(self, labels: Sequence[str]) -> bool:
        """The full 7-rank path (domain..species) is a reference path."""
        return tuple(labels[:7]) in self.paths

    def genera_of(self, species: str) -> Set[str]:
        """Genus member names spanned by a species label (group members resolved one by one; a member unknown to the
        reference falls back to the first word of its name)."""
        out: Set[str] = set()
        for m in members(species):
            gs = self.member_genera.get(m)
            out |= gs if gs else {m.split(" ", 1)[0]}
        return out

    def genus_of_member(self, name: str) -> str:
        """The reference genus of one species member name when unique ('' otherwise)."""
        gs = self.member_genera.get(name)
        return next(iter(gs)) if gs and len(gs) == 1 else ""

    def is_placeholder(self, name: str) -> bool:
        """Placeholder test with the member's reference genus (so 'Candidatus X y' or '[Eubacterium] ... group z'
        are judged on their epithet)."""
        return is_placeholder(name, self.genus_of_member(name))

    def is_pair(self, genus: str, species: str) -> bool:
        return (genus, species) in self.pairs

    def species_taxon(self, genus: str, names: Iterable[str]) -> str:
        """The species label under ``genus`` whose member set equals ``names``, or ''."""
        return self._by_set.get((genus, frozenset(names)), "")

    def genus_path(self, genus: str) -> Optional[Tuple[str, ...]]:
        """Most frequent upper path (domain..family) of a genus label; None if the genus is not in the reference."""
        c = self.genus_paths.get(genus)
        if not c:
            return None
        return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def lineages_from_fasta(path: str) -> Dict[str, List[str]]:
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                h = line[1:].rstrip("\n")
                if "\t" in h:
                    sid, tax = h.split("\t", 1)
                else:
                    sid, _, tax = h.partition(" ")
                out[sid] = parse_lineage(tax)
    return out


# ------------------------------------------------------------------------------------------------ species rule

_TOKEN_SPLIT = re.compile(r"[-_ \[\]]+")


def genus_token_agrees(species: str, genus: str, synonyms: Optional[Mapping[str, Iterable[str]]] = None) -> bool:
    """The genus part of a species name agrees with the genus label (a reference record can carry a
    source-organism species name from another genus).

    True when the species name starts with a genus member followed by a space (Greengenes2 suffixes included, as
    written), or when its first word (brackets and 'Candidatus' removed) is one of the words of a genus member
    split at '-', '_', spaces and brackets (composite genera such as 'Escherichia-Shigella' or
    '[Ruminococcus] torques group'). 'Salmonella enterica' under 'Escherichia-Shigella' disagrees.

    ``synonyms`` maps a genus label (or group member) to further genus names its species may carry, for
    references that keep a renamed species under the old genus field (Greengenes2: 'Segatella copri' under
    'Prevotella'). Only declared pairs are accepted."""
    if not species or not genus:
        return False
    w = species.split()
    if w and w[0] == "Candidatus" and len(w) > 1:
        w = w[1:]
    first = w[0].strip("[]") if w else ""
    for g in members(genus):
        if species.startswith(g + " "):
            return True
        if first and first in set(_TOKEN_SPLIT.split(g)) - {"", "Candidatus"}:
            return True
        if synonyms and first:
            for alias in synonyms.get(g, ()):
                if species.startswith(alias + " ") or first == alias:
                    return True
    return False



def resolve_species(species: str, genus: str, ref: RefTaxonomy,
                    alternatives: Sequence[str] = (), upper: Sequence[str] = (), genus_token: bool = True,
                    genus_synonyms: Optional[Mapping[str, Iterable[str]]] = None) -> Tuple[str, str]:
    """Apply the cross-genus group rule, rank-path consistency and the placeholder rule to one species call.

    ``species``: the species label proposed for the query ('' for none); ``genus``: the genus call that stands
    ('' for none); ``alternatives``: further species candidates in order of preference (for the placeholder rule);
    ``upper``: the called domain..family labels; when all five are given, path consistency is judged on the full
    rank path (labels keyed by path, so a name used under two paths is two taxa), otherwise on the (genus, species)
    pair.
    Returns ``(label, reason)``; label '' means the species rank abstains. Reasons: 'empty', 'no_genus', 'kept',
    'cross_genus_subset', 'cross_genus_abstain', 'genus_token_subset', 'genus_token_abstain', 'path_inconsistent',
    'placeholder_trimmed', 'placeholder_alternative', 'placeholder_abstain'.

    ``genus_token=False`` switches the genus-token check off (the cross-genus, path and placeholder rules stay on);
    ``genus_synonyms`` declares accepted genus aliases for the token check (see :func:`genus_token_agrees`)."""
    def tok_ok(m, g):
        return (not genus_token) or genus_token_agrees(m, g, genus_synonyms)

    if is_nocall(species):
        return "", "empty"
    if is_nocall(genus):
        return "", "no_genus"
    full = len(upper) == 5 and all(not is_nocall(x) for x in upper)

    def on_path(lab):
        if full:
            return ref.is_path(list(upper) + [genus, lab])
        return ref.is_pair(genus, lab)

    def valid(lab):
        ms = members(lab)
        return bool(lab) and len(ref.genera_of(lab)) == 1 and all(tok_ok(m, genus) for m in ms) \
            and not any(ref.is_placeholder(m) for m in ms) and on_path(lab)

    mem = members(species)
    named = [m for m in mem if not ref.is_placeholder(m)]
    if len(named) < len(mem):
        if named:
            lab = ref.species_taxon(genus, named)
            if lab and valid(lab):
                return lab, "placeholder_trimmed"
        for alt in alternatives:
            if not is_nocall(alt) and valid(alt):
                return alt, "placeholder_alternative"
        return "", "placeholder_abstain"
    gm = set(members(genus))
    span = ref.genera_of(species)
    tok = [m for m in mem if tok_ok(m, genus)]
    if len(span) > 1 or len(tok) < len(mem):
        kind = "cross_genus" if len(span) > 1 else "genus_token"
        inside = [m for m in mem if (ref.genera_of(m) & gm) and tok_ok(m, genus)]
        if inside and len(inside) < len(mem):
            lab = ref.species_taxon(genus, inside)
            if lab and valid(lab):
                return lab, kind + "_subset"
        return "", kind + "_abstain"
    if not on_path(species):
        return "", "path_inconsistent"
    return species, "kept"


def apply_species_rule(labels: Sequence[str], ref: RefTaxonomy, alternatives: Sequence[str] = (),
                       genus_token: bool = True,
                       genus_synonyms: Optional[Mapping[str, Iterable[str]]] = None) -> Tuple[List[str], str]:
    """7 labels (a gated lineage, '' = no call) -> 7 labels with the species rank resolved; ranks above unchanged."""
    lab = [("" if is_nocall(x) else x) for x in labels]
    if not lab[SP]:
        return lab, "empty"
    s, why = resolve_species(lab[SP], lab[G], ref, alternatives, upper=lab[:G], genus_token=genus_token,
                             genus_synonyms=genus_synonyms)
    lab[SP] = s
    return lab, why


def path_problems(labels: Sequence[str], ref: RefTaxonomy) -> List[str]:
    """Rank-path checks of one called lineage: 'species_not_in_genus', 'species_cross_genus',
    'species_placeholder', 'genus_path_mismatch' (the called upper ranks are not the genus's reference path),
    'gap' (a call below a rank with no call)."""
    lab = [("" if is_nocall(x) else x) for x in labels]
    out = []
    seen_gap = False
    for x in lab:
        if not x:
            seen_gap = True
        elif seen_gap:
            out.append("gap")
            break
    g, s = lab[G], lab[SP]
    if s:
        if g and not ref.is_pair(g, s):
            out.append("species_not_in_genus")
        elif g and all(lab[:G]) and not ref.is_path(lab):
            out.append("species_path_not_in_reference")
        if g and not all(genus_token_agrees(m, g) for m in members(s)):
            out.append("species_genus_token")
        if len(ref.genera_of(s)) > 1:
            out.append("species_cross_genus")
        if any(ref.is_placeholder(m) for m in members(s)):
            out.append("species_placeholder")
    if g:
        paths = ref.genus_paths.get(g)
        called = tuple(lab[:G])
        if paths and all(called[i] for i in range(G)) and called not in paths:
            out.append("genus_path_mismatch")
    return out


# ------------------------------------------------------------------------------------------------ ambiguity runs

_AMBIG = re.compile(r"[^ACGTUacgtu]+")


def max_ambiguity_run(seq: str) -> int:
    """Length of the longest run of characters other than A, C, G, T/U (N and the IUPAC ambiguity codes, gaps)."""
    return max((len(m) for m in _AMBIG.findall(seq)), default=0)


def ambiguous_records(records: Iterable[Tuple[str, str]], min_run: int = 5) -> Set[str]:
    """Ids of reference records with an ambiguity run of at least ``min_run`` characters."""
    return {rid for rid, seq in records if max_ambiguity_run(seq) >= min_run}


def drop_hits_to(hits: Sequence[tuple], excluded: Set[str]) -> List[tuple]:
    """Hits ``(target, identity, ...)`` without those whose target is excluded (order kept)."""
    return [h for h in hits if h[0] not in excluded]


# ------------------------------------------------------------------------------------------------ genus decision


def knn_genus_vote(hits: Sequence[Tuple[str, float]], genus_of: Dict[str, str], delta: float = 0.01,
                   tau: float = 0.005, k: int = 64) -> Tuple[str, float, float]:
    """Weighted vote over the nearest labelled neighbours.

    ``hits``: (target, identity) sorted by identity descending; ``genus_of``: target -> genus label ('' = none).
    Neighbours with identity >= top - ``delta`` (at most ``k``) vote with weight exp(-(top - identity) / tau); a
    group genus label splits its weight equally over its members. Returns (genus member with the largest share,
    its share, top identity); ('', 0, top) when no labelled neighbour exists. Ties go to the alphabetically first
    name (deterministic)."""
    lab = [(t, i) for t, i in hits if genus_of.get(t)]
    if not lab:
        return "", 0.0, (hits[0][1] if hits else 0.0)
    top = lab[0][1]
    w: Dict[str, float] = defaultdict(float)
    tot = 0.0
    for t, i in lab[:k]:
        if i < top - delta:
            break
        wt = math.exp(-(top - i) / tau) if tau > 0 else 1.0
        gm = members(genus_of[t])
        for m in gm:
            w[m] += wt / len(gm)
        tot += wt
    best = sorted(w.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return best[0], best[1] / tot, top


def agreement_genus(net_genus: str, net_conf: float, nn_genus: str, nn_share: float, nn_ident: float,
                    t_agree: float, t_net: float, t_nn: float, s_nn: float, t_id_agree: float = math.inf) -> Tuple[str, str]:
    """Network-versus-neighbour agreement rule. Returns (genus, source).

    * agree (the network genus contains the neighbour genus): call it when net_conf >= t_agree or
      nn_ident >= t_id_agree;
    * disagree: the network genus when net_conf >= t_net; else the neighbour genus when nn_ident >= t_nn and
      nn_share >= s_nn; else no call.
    A missing side counts as disagreement with an absent candidate."""
    ng = "" if is_nocall(net_genus) else net_genus
    if ng and nn_genus and nn_genus in members(ng):
        if net_conf >= t_agree or nn_ident >= t_id_agree:
            return ng, "agree"
        return "", "agree_low"
    if ng and net_conf >= t_net:
        return ng, "net"
    if nn_genus and nn_ident >= t_nn and nn_share >= s_nn:
        return nn_genus, "nn"
    return "", "none"


def lineage_for_genus(genus: str, net_labels: Sequence[str], ref: RefTaxonomy) -> List[str]:
    """Upper ranks for a genus decision: the network's own labels when they are the genus's reference path,
    otherwise the genus's most frequent reference path (so the called lineage is consistent). Species is ''."""
    lab = [("" if is_nocall(x) else x) for x in net_labels][:G] + ["", ""]
    if not genus:
        return lab[:G] + ["", ""]
    paths = ref.genus_paths.get(genus)
    if paths and tuple(lab[:G]) in paths:
        return lab[:G] + [genus, ""]
    p = ref.genus_path(genus)
    if p is None:
        return lab[:G] + [genus, ""]
    return list(p) + [genus, ""]


# ------------------------------------------------------------------------------------------------ classify hook


def ref_from_hierarchy(hierarchy) -> RefTaxonomy:
    """RefTaxonomy from a checkpoint's taxonomy hierarchy (a TaxonomyHierarchy or a dict holding
    ``child_to_parent``: {(level, name): (parent_level, parent_name)}). Each species is walked up to the domain."""
    c2p = hierarchy.get("child_to_parent") if isinstance(hierarchy, dict) else getattr(hierarchy, "child_to_parent")
    lineages = []
    for (level, name) in list(c2p):
        if level != "species":
            continue
        lab = [""] * 7
        key = (level, name)
        seen = 0
        while key in c2p or key[0] in RANKS:
            if key[0] in RANKS:
                lab[RANKS.index(key[0])] = "" if is_nocall(key[1]) else key[1]
            if key not in c2p or seen > 10:
                break
            key = c2p[key]
            seen += 1
        lineages.append(lab)
    return RefTaxonomy(lineages)


RULE_KEYS = ("genus_token", "genus_synonyms")


def rule_options(entry=None, override: str = "auto") -> dict:
    """Keyword options of the species rule from a checkpoint's ``label_rules`` entry
    (``{"genus_token": bool, "genus_synonyms": {genus: [alias, ...]}}``; both optional) and the command-line
    ``--genus-token-check`` value (``auto`` keeps the checkpoint's choice, ``on`` / ``off`` override it).
    No entry means the genus-token check is on. A malformed entry raises ValueError."""
    out = {"genus_token": True, "genus_synonyms": None}
    if entry is not None:
        if not isinstance(entry, dict) or set(entry) - set(RULE_KEYS):
            raise ValueError(f"label_rules must be a dict with keys {RULE_KEYS}: {entry!r}")
        if "genus_token" in entry:
            if not isinstance(entry["genus_token"], bool):
                raise ValueError("label_rules.genus_token must be true or false")
            out["genus_token"] = entry["genus_token"]
        syn = entry.get("genus_synonyms")
        if syn is not None:
            if not isinstance(syn, dict) or not all(isinstance(k, str) and isinstance(v, (list, tuple, set))
                                                    and all(isinstance(a, str) for a in v) for k, v in syn.items()):
                raise ValueError("label_rules.genus_synonyms must map a genus name to a list of names")
            out["genus_synonyms"] = {k: set(v) for k, v in syn.items()}
    if override not in ("auto", "on", "off"):
        raise ValueError(f"--genus-token-check must be auto, on or off, not {override!r}")
    if override != "auto":
        out["genus_token"] = override == "on"
    return out


def apply_to_predictions(predictions: dict, ref: RefTaxonomy, genus_token: bool = True,
                         genus_synonyms: Optional[Mapping[str, Iterable[str]]] = None) -> Tuple[dict, Counter]:
    """Species rule on a predictor output dict ({'predictions': {level: [labels]}, 'confidence': {level: [..]}}),
    in place. The genus label stands; a species that fails the rule becomes 'unknown' with confidence 0.0, so the
    writer reports it as unclassified. Returns (predictions, Counter of reasons)."""
    why = Counter()
    sp = predictions["predictions"]["species"]
    ge = predictions["predictions"]["genus"]
    cs = predictions["confidence"]["species"]
    for i in range(len(sp)):
        s, w = resolve_species(sp[i] or "", ge[i] or "", ref, genus_token=genus_token, genus_synonyms=genus_synonyms)
        why[w] += 1
        if s != sp[i] and not (is_nocall(s) and is_nocall(sp[i])):
            sp[i] = s if s else "unknown"
            if not s:
                cs[i] = 0.0
    return predictions, why
