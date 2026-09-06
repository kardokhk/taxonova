"""Recommended per-rank confidence profile, and how a single value scales it.

One definition, imported by the command-line parser, by the TSV writer that
applies the gates, and by the hosted web app. Keeping it here rather than in
either caller is what stops the two entry points drifting apart: they did, and
Reviewer 2 found it, with the web page opening at 0.5 while the manuscript
recommended 0.7 at genus and 0.8 at species.

Deliberately stdlib-only, so `--help` still works in an environment where the
deep-learning stack is missing or broken.
"""

#: 0.5 for domain through family, 0.7 for genus. Reported in Methods,
#: "Architecture".
GRADED_PROFILE = {
    "domain": 0.5,
    "phylum": 0.5,
    "class": 0.5,
    "order": 0.5,
    "family": 0.5,
    "genus": 0.7,
}

#: Species gate the profile is expressed against, and the shipped default.
GRADED_SPECIES = 0.8


def scaled_profile(confidence_threshold):
    """The parent gates for a requested species confidence.

    The profile scales in its own ratio, so one requested number means the same
    thing at every rank. Requesting 0.0 opens every gate, which is what a
    post-hoc threshold sweep needs from a single inference pass; requesting the
    default reproduces GRADED_PROFILE exactly.
    """
    scale = confidence_threshold / GRADED_SPECIES if GRADED_SPECIES else 1.0
    return {lvl: min(1.0, gate * scale) for lvl, gate in GRADED_PROFILE.items()}
