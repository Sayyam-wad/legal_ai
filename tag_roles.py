"""Rule-based rhetorical role tagging.

What changed vs. the first version
  * Roles are assigned per *sentence*, and a chunk gets every role any of its
    sentences carries (multi-label). Previously a whole 512-word chunk got one
    label from the first matching phrase, so a chunk with one "sentence of"
    anywhere became RPC even if it was mostly facts.
  * Patterns are regexes with word boundaries ("act" no longer matches "fact").
  * Position-aware: the opening chunk is the PREAMBLE, outcome language in the
    final chunks is RPC.

Output: data/chunks/chunks_tagged.csv with
  roles  pipe-separated list of every role found in the chunk (e.g. "FAC|STA")
  role   the single highest-priority role (kept for backwards compatibility)

Run: python tag_roles.py
"""

import re

import pandas as pd
from tqdm import tqdm

import config
from legal_text import split_sentences


def _rx(*phrases: str) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(phrases) + r")\b", re.I)


# Order = priority used for the single `role` column.
ROLE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("RPC", _rx(
        r"(appeals?|petitions?|applications?) (is|are|stands?|shall stand|be) (accordingly |hereby )?(dismissed|allowed|partly allowed|disposed of)",
        r"we (accordingly |therefore |hereby )?(allow|dismiss) (this|the|these) (appeals?|petitions?)",
        r"(conviction|sentence|order|judgment) (and sentence )?(is|are|stands?) (hereby )?(set aside|confirmed|upheld|maintained|modified|reduced)",
        r"bail is (granted|rejected|refused|cancelled)", r"(is|are) (hereby )?acquitted",
        r"(is|are) sentenced to", r"we (hereby )?(order|direct) that", r"no order as to costs",
        r"the (appellants?|accused) (shall|be) (released|set at liberty)",
    )),
    ("RATIO", _rx(
        r"we are (of the |clearly of the )?(view|opinion) that", r"in our (considered )?(view|opinion)",
        r"it is (well )?settled( law)? that", r"the (settled )?principle (is|that)",
        r"(thus|therefore|accordingly),? we hold", r"we (therefore )?hold that",
        r"for the (foregoing|aforesaid|reasons stated)", r"we find (no )?(merit|substance|force)",
        r"we are (not )?(satisfied|unable to (accept|agree))", r"the (legal )?position (is|that) ",
        r"(it|this) (must|would) follow(s)? that", r"the test (is|to be applied)",
    )),
    ("ISSUE", _rx(
        r"the (only |short |sole |main |next )?(question|issue|point)s? (that|which) (arises?|falls?|requires?)",
        r"the (question|issue|point)s? for (our )?(consideration|determination|decision)",
        r"(points?|questions?) (that|which) arise for", r"whether (or not )?the (accused|appellants?|respondents?|petitioners?|high court|trial court)",
        r"the point for determination", r"we are called upon to decide",
    )),
    ("PRE_NOT_RELIED", _rx(
        r"(is|are) not applicable to the facts", r"cannot be relied (up)?on", r"(is|are) (clearly )?distinguishable",
        r"(does|do) not (help|assist|advance)", r"(is|are) of no (assistance|help)",
        r"has no application", r"(ratio|decision) .{0,40} does not apply",
    )),
    ("PRE_RELIED", _rx(
        r"relied (up)?on by", r"we (may )?place reliance", r"as (held|observed|laid down) (by this court )?in",
        r"following the (ratio|decision|principle) in", r"in view of the (decision|judgment|law laid down) in",
        r"this court in .{0,60}(held|observed)", r"reference (may|can) be made to",
        r"(AIR|SCC|SCR) \d", r"\(\d{4}\) \d+ SCC",
    )),
    ("ARG_PETITIONER", _rx(
        r"(learned )?(senior )?counsel (appearing )?(for|on behalf of) the (appellants?|petitioners?|accused) (submitted|contended|urged|argued|pointed out)",
        r"(it was|it is) (contended|submitted|urged|argued) (by|on behalf of) the (appellants?|petitioners?|accused)",
        r"the (appellants?|petitioners?) contends?", r"on behalf of the (appellants?|petitioners?),? it (was|is)",
    )),
    ("ARG_RESPONDENT", _rx(
        r"(learned )?(senior )?counsel (appearing )?(for|on behalf of) the (respondents?|state|prosecution|union) (submitted|contended|urged|argued|pointed out)",
        r"(learned )?(additional )?(solicitor general|public prosecutor|standing counsel) (submitted|contended|urged|argued)",
        r"(it was|it is) (contended|submitted|urged|argued) (by|on behalf of) the (respondents?|state|prosecution)",
        r"the (respondents?|state|prosecution) (contends?|submits?)",
    )),
    ("RLC", _rx(
        r"the (learned )?(trial court|trial judge|high court|sessions (court|judge)|additional sessions judge|lower court|appellate court|tribunal|magistrate|division bench|single judge)"
        r" (held|found|observed|convicted|acquitted|came to the conclusion|took the view|dismissed|allowed|rejected)",
        r"the impugned (judgment|order)", r"(convicted|acquitted) by the (trial|sessions|high) court",
    )),
    ("FAC", _rx(
        r"(fir|f\.i\.r\.|first information report) (was|has been) (lodged|registered|filed|recorded)",
        r"the (prosecution|complainant)'?s? (case|story|version) (is|was)", r"briefly (stated|put),? the facts",
        r"the (relevant |material |brief )?facts (of the case |leading to .{0,40})?(are|were|is)",
        r"it is alleged that", r"the deceased (was|had)", r"the incident (took place|occurred|happened)",
        r"on the (night|morning|evening|date) of", r"was (arrested|married|appointed|employed)",
    )),
    ("ANALYSIS", _rx(
        r"we have (carefully )?(considered|perused|gone through|examined)", r"on (a )?(careful )?(perusal|examination|reading|scrutiny) of",
        r"having (regard|considered|heard)", r"the evidence (on record )?(shows|establishes|discloses|indicates)",
        r"there is (no|nothing|sufficient|ample) (evidence|material)", r"the (court|we) (note|observe)s? that",
    )),
    ("STA", _rx(
        r"(under|u/s\.?|of) (sections?|ss?\.|articles?) \d+", r"sections? \d+[A-Z]? (of the |of |read with )",
        r"article \d+[A-Z]? of the constitution", r"(i\.?p\.?c|cr\.?p\.?c|indian penal code|code of criminal procedure|evidence act)\b",
        r"the [A-Z][\w ]{2,60} act,? (19|20)\d{2}",
    )),
]
PRIORITY = [name for name, _ in ROLE_PATTERNS]
PREAMBLE_RX = _rx(r"supreme court of india", r"the judgment was delivered by", r"(criminal|civil) appeal no",
                  r"writ petition", r"special leave petition", r"appeal \(cr?l?\.?\)")


def sentence_roles(sentence: str) -> set[str]:
    return {name for name, rx in ROLE_PATTERNS if rx.search(sentence)}


def tag_chunk(text: str, chunk_idx: int, n_chunks: int, doc_type: str) -> list[str]:
    if doc_type == "statute":
        return ["STA"]
    roles: set[str] = set()
    for s in split_sentences(text):
        roles |= sentence_roles(s)
    if chunk_idx == 0 and PREAMBLE_RX.search(text):
        roles.add("PREAMBLE")
    # Outcome language only counts as the final ruling near the end of a judgment.
    if "RPC" in roles and n_chunks > 3 and chunk_idx < n_chunks - 2:
        roles.discard("RPC")
    return sorted(roles, key=lambda r: PRIORITY.index(r) if r in PRIORITY else len(PRIORITY))


def main():
    print("Loading chunks...")
    df = pd.read_csv(config.CHUNKS_CSV)
    print(f"Total chunks: {len(df)}")

    tqdm.pandas(desc="Tagging")
    df["roles"] = df.progress_apply(
        lambda r: "|".join(tag_chunk(str(r["text"]), int(r["chunk_idx"]), int(r["n_chunks"]), r["doc_type"])),
        axis=1)
    df["role"] = df["roles"].apply(lambda s: s.split("|")[0] if s else "NONE")

    exploded = df["roles"].str.split("|").explode()
    exploded = exploded[exploded != ""]
    print("\nChunks containing each role (multi-label):")
    print(exploded.value_counts().to_string())
    coverage = (df["roles"] != "").mean() * 100
    print(f"\nCoverage (chunks with >=1 role): {coverage:.1f}%")

    df.to_csv(config.TAGGED_CSV, index=False)
    print(f"Saved to: {config.TAGGED_CSV.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
