#!/usr/bin/env bash
# Build Paper 1 from its section files.
#
#   scripts/build_paper.sh
#
# Joins docs/paper/{abstract,intro,related,methods,results,discussion,backmatter}.md,
# in that order, into docs/paper/build/paper.md, then runs pandoc to make
# build/paper.html (standalone, figures embedded) and build/paper.docx.
# No LaTeX engine is installed, so no PDF.
#
# While joining:
#   - each file's italic "*Draft, ...*" note (first paragraph starting
#     "*Draft") is dropped;
#   - abstract.md's "# Title options" section is dropped, and the option
#     marked *Recommended* becomes the document title;
#   - relative image paths are rewritten to resolve from build/.
# HTML source comments stay in paper.md; pandoc strips them from the output.
# The section text itself is never edited.
#
# Prints pandoc's warnings (unresolved citations, missing images), then
# TODO(...) counts and word counts per section. Word counts are of the
# section as built (draft note and HTML comments removed, tables included,
# each unrendered [@key] group counted as a word).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PAPER=$REPO/docs/paper
BUILD=$PAPER/build
SECTIONS=(abstract intro related methods results discussion backmatter)

command -v pandoc >/dev/null || { echo "build_paper: pandoc not found" >&2; exit 1; }
# --embed-resources arrived in pandoc 2.19; older versions call it --self-contained.
if pandoc --help | grep -q -- '--embed-resources'; then
    EMBED=--embed-resources
else
    EMBED=--self-contained
fi

# Title: the bold text of the numbered option marked *Recommended* in
# abstract.md's "# Title options" section.
TITLE=$(perl -0777 -ne '
    /^# Title options[ \t]*\n(.*?)(?=^# |\z)/ms or die "build_paper: no \"# Title options\" section in abstract.md\n";
    for my $item (split /^(?=\d+\.\s)/m, $1) {
        next unless $item =~ /\*Recommended\*/;
        $item =~ /\*\*(.+?)\*\*/s or die "build_paper: the *Recommended* option has no **title**\n";
        (my $t = $1) =~ s/\s+/ /g;
        print $t;
        exit;
    }
    die "build_paper: no option marked *Recommended* in abstract.md\n";
' "$PAPER/abstract.md")

rm -rf "$BUILD"
mkdir -p "$BUILD/sections"

i=0
for s in "${SECTIONS[@]}"; do
    i=$((i + 1))
    src=$PAPER/$s.md
    [[ -f $src ]] || { echo "build_paper: missing $src" >&2; exit 1; }
    awk -v file="$s.md" '
        /^# Title options[ \t]*$/ { drop = 1; next }      # abstract.md only
        drop && /^# /             { drop = 0 }
        drop                      { next }
        !done && /^\*Draft/       { skip = 1 }
        skip {
            if ($0 ~ /^[ \t]*$/) { skip = 0; done = 1 } else last = $0
            next
        }
        { print }
        END {
            if (skip) { print "build_paper: " file ": draft note runs to end of file" > "/dev/stderr"; exit 1 }
            if (done && last !~ /\*[ \t]*$/)
                print "build_paper: " file ": dropped draft note does not end in *; check what was cut" > "/dev/stderr"
        }
    ' "$src" > "$BUILD/sections/$(printf '%02d' "$i")-$s.md"
done

# Assemble: YAML title, then the sections separated by blank lines, with
# relative image paths (written relative to docs/paper/) moved up one level.
{
    printf -- "---\ntitle: '%s'\nlang: en-GB\nlink-citations: true\nreference-section-title: References\n---\n\n" \
        "${TITLE//\'/\'\'}"
    for f in "$BUILD"/sections/*.md; do
        cat "$f"
        printf '\n\n'
    done
} | perl -0777 -pe 's{(!\[[^\]]*\]\()(?![a-zA-Z][a-zA-Z0-9+.-]*:|/|#)}{$1../}g' > "$BUILD/paper.md"

# Report images that will not resolve (pandoc also warns).
perl -0777 -ne 'print "$1\n" while /!\[[^\]]*\]\(([^)\s]+)/g' "$BUILD/paper.md" |
    while read -r img; do
        [[ $img == *:* || -f $BUILD/$img ]] || echo "build_paper: missing image $img" >&2
    done

cd "$BUILD"
COMMON=(--from=markdown --strip-comments --citeproc
        --bibliography="$PAPER/refs.bib" --resource-path=.)
echo "== pandoc: paper.html"
pandoc "${COMMON[@]}" --standalone "$EMBED" --to=html5 --output=paper.html paper.md
echo "== pandoc: paper.docx"
pandoc "${COMMON[@]}" --to=docx --output=paper.docx paper.md

echo "== built"
echo "title: $TITLE"
ls -l paper.md paper.html paper.docx

echo "== TODO tags left (section text as built, HTML comments included)"
for f in sections/*.md; do
    printf '%-24s' "$(basename "$f")"
    { grep -o 'TODO([a-z]*)' "$f" || true; } | sort | uniq -c | awk '{printf " %s=%s", $2, $1} END {print ""}'
done

echo "== words per section"
total=0
for f in sections/*.md; do
    n=$(pandoc --from=markdown --strip-comments --to=plain "$f" | wc -w)
    total=$((total + n))
    printf '%-24s %6d\n' "$(basename "$f")" "$n"
done
printf '%-24s %6d\n' total "$total"
