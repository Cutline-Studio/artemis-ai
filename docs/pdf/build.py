"""Builds the Artemis PDFs from the Markdown docs: python docs/pdf/build.py (needs `markdown` and Playwright)."""
import datetime
import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BLUEPRINT = Path("/home/user/artm-blueprint")
TODAY = datetime.date.today().strftime("%B %-d, %Y")

CSS = """
@page { size: Letter; margin: 0.8in 0.75in 0.85in; }
* { box-sizing: border-box; }
body { font-family: 'Inter', 'Helvetica Neue', 'DejaVu Sans', Arial, sans-serif; color: #121317; font-size: 10.5pt; line-height: 1.5; }
.cover { height: 9.2in; display: flex; flex-direction: column; justify-content: center; page-break-after: always; }
.cover .mark { width: 64px; height: 64px; color: #3279f9; }
.cover h1 { font-size: 34pt; letter-spacing: -0.02em; margin: 24px 0 8px; border: 0; font-weight: 600; }
.cover p { font-size: 13pt; color: #45474d; margin: 4px 0; max-width: 5.5in; }
.cover .meta { margin-top: 36px; font-size: 10pt; color: #6a6a71; }
.toc { page-break-after: always; }
.toc h2 { border: 0; }
.toc ol { font-size: 12pt; line-height: 2; }
section.doc { page-break-before: always; }
h1 { font-size: 22pt; letter-spacing: -0.01em; margin: 0 0 14px; padding-bottom: 8px; border-bottom: 2px solid #3279f9; }
h2 { font-size: 15pt; margin: 22px 0 8px; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 16px 0 6px; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
a { color: #1f5fd6; text-decoration: none; }
code { font-family: 'JetBrains Mono', 'DejaVu Sans Mono', monospace; font-size: 8.8pt; background: #f1f3f8; padding: 1px 4px; border-radius: 4px; }
pre { background: #f6f7fa; border: 1px solid #e3e6ec; border-radius: 8px; padding: 10px 12px; white-space: pre-wrap; word-break: break-word; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.4pt; line-height: 1.45; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 14px; font-size: 9pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th { background: #eef3ff; text-align: left; font-weight: 600; }
th, td { border: 1px solid #dfe3ea; padding: 5px 7px; vertical-align: top; }
td code { word-break: break-word; }
blockquote { border-left: 3px solid #3279f9; margin: 8px 0; padding: 2px 12px; color: #45474d; }
.note { background: #fff8e6; border: 1px solid #f1d58a; border-radius: 8px; padding: 8px 12px; font-size: 9.5pt; }
"""
MARK = ('<svg class="mark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 20 12 3l8 17"/>'
        '<path d="M7.5 13h9"/><circle cx="12" cy="3" r="1.1" fill="currentColor"/></svg>')


def md(path: Path, note: str = "") -> str:
    text = path.read_text()
    text = re.sub(r"\]\((?!https?:)([^)]+\.md)\)", r"](#)", text)  # cross-file links point inside the PDF bundle
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    body = re.sub(r"<td>(\d{4}-\d{2}-\d{2})</td>", r'<td style="white-space:nowrap">\1</td>', body)
    return f'<section class="doc">{note}{body}</section>'


def page(title: str, subtitle: str, sections: list[tuple[str, str]], audience: str) -> str:
    toc = "".join(f"<li>{name}</li>" for name, _ in sections)
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{CSS}</style></head><body>"
            f"<div class='cover'>{MARK}<h1>{title}</h1><p>{subtitle}</p>"
            f"<p class='meta'>Artemis AI · {TODAY} · {audience}</p></div>"
            f"<div class='toc'><h2>Contents</h2><ol>{toc}</ol></div>" + "".join(html for _, html in sections) + "</body></html>")


def render(html: str, pdf: Path):
    src = OUT / (pdf.stem + ".html")
    src.write_text(html)
    js = f"""
const {{ chromium }} = require('/opt/node22/lib/node_modules/playwright');
(async () => {{
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://{src}', {{ waitUntil: 'load' }});
  await p.pdf({{ path: '{pdf}', format: 'Letter', printBackground: true, displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size:8px;color:#888;width:100%;padding:0 0.75in;display:flex;justify-content:space-between;"><span>{pdf.stem.replace("-", " ")}</span><span><span class=pageNumber></span> / <span class=totalPages></span></span></div>',
    margin: {{ top: '0.7in', bottom: '0.8in', left: '0.75in', right: '0.75in' }} }});
  await b.close();
}})();"""
    subprocess.run(["node", "-e", js], check=True)
    src.unlink()
    print(pdf, pdf.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    docs = ROOT / "docs"
    guide = ("Developer starter guide", md(docs / "DEVELOPER_GUIDE.md"))
    render(page("Artemis AI Blueprint", "Decisions, architecture, the neural core, and the owner's Azure capacity request.", [
        ("Blueprint: decisions and brain map", md(BLUEPRINT / "BLUEPRINT.md")),
        ("Architecture: the application", md(docs / "ARCHITECTURE.md")),
        ("Neural core: the model", md(docs / "NEURAL_CORE.md")),
        guide,
        ("Appendix: Azure capacity request", md(BLUEPRINT / "azure-sales-request.md",
                                                 "<p class='note'>Owner's file. Contains the subscription ID and contact details; "
                                                 "share this PDF only with people who should see them.</p>")),
    ], "Owner copy, includes account details"), OUT / "Artemis-Blueprint.pdf")
    render(page("Artemis AI Developer Starter Guide", "Set up, run, train, ship. Everything a new developer needs on day one.", [
        guide,
        ("Architecture: the application", md(docs / "ARCHITECTURE.md")),
        ("Neural core: the model", md(docs / "NEURAL_CORE.md")),
    ], "For developers, safe to share"), OUT / "Artemis-Developer-Starter-Guide.pdf")
