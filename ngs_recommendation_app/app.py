#!/usr/bin/env python3
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = os.path.dirname(__file__)


class RecommendationHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._serve_file("index.html", "text/html; charset=utf-8")
            return
        if path == "/styles.css":
            self._serve_file("styles.css", "text/css; charset=utf-8")
            return
        if path == "/app.js":
            self._serve_file("app.js", "application/javascript; charset=utf-8")
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/recommend":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body or "{}")
        except Exception as exc:
            self._send_json(400, {"error": f"Invalid JSON: {exc}"})
            return

        self._send_json(200, build_recommendation(payload))

    def _serve_file(self, filename, content_type):
        file_path = os.path.join(ROOT, filename)
        if not os.path.exists(file_path):
            self._send_json(404, {"error": "File not found"})
            return
        with open(file_path, "rb") as handle:
            content = handle.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, code, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


def build_recommendation(payload):
    purpose = payload.get("purpose", "clinical")
    sample = payload.get("sample", "blood")
    urgency = payload.get("urgency", "standard")
    sensitivity = payload.get("sensitivity", "balanced")
    complexity = payload.get("complexity", "oncology")
    team = payload.get("team", "hybrid")

    score = 0.0
    if purpose == "clinical":
        score += 0.35
    elif purpose == "discovery":
        score += 0.25
    else:
        score += 0.3

    if sample == "ffpe":
        score += 0.15
    elif sample == "tissue":
        score += 0.12
    elif sample == "blood":
        score += 0.1
    else:
        score += 0.08

    if urgency == "urgent":
        score += 0.2
    elif urgency == "standard":
        score += 0.1
    else:
        score += 0.12

    if sensitivity == "high":
        score += 0.2
    elif sensitivity == "balanced":
        score += 0.1
    else:
        score += 0.08

    if complexity == "metagenomic":
        score += 0.12
    elif complexity == "rna":
        score += 0.1
    else:
        score += 0.08

    if team == "bioinformatics":
        score += 0.08
    elif team == "wetlab":
        score += 0.05
    else:
        score += 0.06

    coherence = round(min(0.99, 0.68 + score * 0.28), 2)
    entropy = round(max(0.1, 0.95 - score * 0.35), 2)

    if purpose == "clinical" and urgency == "urgent":
        workflow = "Rapid targeted oncology panel with UMI-aware QC"
        platform = "Illumina NextSeq 2000"
        qc = "Dual-index QC, contamination screen, and coverage gating"
        analysis = "DRAGEN/fastp + variant annotation + clinical report template"
        reporting = "Variant call narrative for molecular tumor boards"
        focus = "Clinical decision support"
    elif complexity == "metagenomic":
        workflow = "Shotgun metagenomics with long-read confirmation"
        platform = "ONT PromethION + Illumina re-sequencing"
        qc = "Host depletion, taxonomic sanity checks, and depth normalization"
        analysis = "Kraken2/MetaPhlAn + assembly + functional profiling"
        reporting = "Microbiome and resistance interpretation report"
        focus = "Environmental and pathogen discovery"
    elif complexity == "rna":
        workflow = "RNA-seq transcriptome workflow with isoform awareness"
        platform = "Illumina NovaSeq X"
        qc = "RIN-aware prep, strandedness validation, and splice QC"
        analysis = "STAR + Salmon + differential expression + pathway enrichment"
        reporting = "Transcriptome atlas and biomarker shortlist"
        focus = "Expression and mechanism"
    else:
        workflow = "Hybrid short-read discovery pipeline with adaptive QC"
        platform = "Illumina NovaSeq X or Element AVITI"
        qc = "Library complexity, duplication, and contamination gates"
        analysis = "BWA-MEM + GATK/DeepVariant + CNV and SV tracking"
        reporting = "Integrated genomic interpretation package"
        focus = "Broad genomic discovery"

    if sample == "ffpe":
        workflow = workflow + " with degraded-sample rescue"
    elif sample == "saliva":
        workflow = workflow + " with low-input adaptation"

    if sensitivity == "high":
        workflow = workflow + " and ultra-sensitive UMI chemistry"

    rationale = (
        f"This recommendation blends wet-lab clinician constraints, bioinformatics feasibility, and biology-oriented interpretation. "
        f"The {focus.lower()} path prioritizes {workflow.lower()} and uses {platform.lower()} to preserve turnaround without sacrificing signal quality."
    )

    return {
        "summary": {
            "title": "Quantum-guided NGS pipeline recommendation",
            "headline": f"{focus} blueprint is ready",
            "description": rationale,
        },
        "recommendation": {
            "workflow": workflow,
            "platform": platform,
            "qc": qc,
            "analysis": analysis,
            "reporting": reporting,
        },
        "qml": {
            "coherence": coherence,
            "entropy": entropy,
            "state": [
                round(min(0.99, coherence + 0.07), 2),
                round(min(0.99, entropy + 0.03), 2),
                round(min(0.99, 0.82 + (0.03 if urgency == "urgent" else 0.01)), 2),
            ],
        },
        "llm_rationale": [
            f"Wet-lab clinicians: prioritize {sample} handling and {urgency} turnaround.",
            f"Bioinformaticians: deploy a stable {analysis.lower()} stack with controlled compute costs.",
            f"Biologists: preserve interpretability through {reporting.lower()}.",
        ],
        "evidence": [
            f"Purpose signal: {purpose}",
            f"Sample handling: {sample}",
            f"Sensitivity profile: {sensitivity}",
            f"Team preference: {team}",
        ],
    }


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8001), RecommendationHandler)
    print("NGS recommendation app running at http://127.0.0.1:8001")
    server.serve_forever()


if __name__ == "__main__":
    main()
