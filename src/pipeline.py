import json
import re
from typing import Any, Dict, List, Optional

import spacy
from transformers import Pipeline, pipeline as hf_pipeline


def _safe_unique(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(normalized)
    return ordered


def _try_load_spacy() -> Optional[spacy.language.Language]:
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        try:
            from spacy.cli import download

            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception:
            return None
    except Exception:
        return None


def _try_hf_pipeline(task: str, model: Optional[str] = None, **kwargs) -> Optional[Pipeline]:
    try:
        return hf_pipeline(task, model=model, **kwargs)
    except Exception:
        return None


class MedicalNLPipeline:
    """
    Local-first NLP pipeline for medical entity extraction, summarization,
    sentiment/intent classification, keyword extraction, and SOAP generation.
    """

    def __init__(self) -> None:
        self.nlp = _try_load_spacy()
        self.sentiment_pipe = _try_hf_pipeline(
            "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.summarizer_pipe = _try_hf_pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            max_length=256,
        )
        self.zero_shot_pipe = _try_hf_pipeline(
            "zero-shot-classification", model="valhalla/distilbart-mnli-12-1"
        )

        self.symptom_terms = [
            "pain",
            "ache",
            "back pain",
            "neck pain",
            "headache",
            "head impact",
            "stiffness",
            "tenderness",
            "discomfort",
            "sleep trouble",
            "mobility issues",
        ]
        self.treatment_terms = [
            "physiotherapy",
            "painkiller",
            "analgesic",
            "therapy",
            "session",
            "exercise",
            "rest",
            "x-ray",
            "follow-up",
        ]
        self.diagnosis_terms = [
            "whiplash",
            "injury",
            "strain",
            "sprain",
            "fracture",
            "concussion",
            "degeneration",
        ]
        self.prognosis_terms = [
            "recovery",
            "full recovery",
            "long-term",
            "six months",
            "improving",
            "worsening",
        ]

    def _doc(self, text: str) -> Optional[spacy.tokens.Doc]:
        if not self.nlp:
            return None
        return self.nlp(text)

    def _keyword_scan(self, text: str, terms: List[str]) -> List[str]:
        found: List[str] = []
        lower = text.lower()
        for term in terms:
            if term.lower() in lower:
                found.append(term)
        return _safe_unique(found)

    def _infer_patient_name(self, text: str) -> Optional[str]:
        match = re.search(r"\b(Ms\.?|Mr\.?)\s+([A-Z][a-z]+)\b", text)
        if match:
            return f"{match.group(1).replace('.', '')} {match.group(2)}"
        return None

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        symptoms = self._keyword_scan(text, self.symptom_terms)
        treatment = self._keyword_scan(text, self.treatment_terms)
        diagnosis = self._keyword_scan(text, self.diagnosis_terms)
        prognosis = self._keyword_scan(text, self.prognosis_terms)

        doc = self._doc(text)
        if doc:
            for ent in doc.ents:
                normalized = ent.text.strip()
                if ent.label_ in {"INJURY", "DISEASE", "DIAGNOSIS"} or any(
                    kw in normalized.lower() for kw in ["injury", "strain", "sprain", "whiplash"]
                ):
                    diagnosis.append(normalized)
                if ent.label_ in {"TREATMENT", "THERAPY"} or "therapy" in normalized.lower():
                    treatment.append(normalized)
                if ent.label_ in {"SYMPTOM"}:
                    symptoms.append(normalized)

        return {
            "symptoms": _safe_unique(symptoms),
            "treatment": _safe_unique(treatment),
            "diagnosis": _safe_unique(diagnosis),
            "prognosis": _safe_unique(prognosis),
        }

    def extract_keywords(self, text: str, top_k: int = 8) -> List[str]:
        doc = self._doc(text)
        if not doc:
            tokens = re.findall(r"[A-Za-z][A-Za-z ]{3,}", text)
            return _safe_unique(tokens)[:top_k]

        candidates: List[str] = []
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            if 3 <= len(phrase) <= 80 and " " in phrase:
                candidates.append(phrase)

        frequencies: Dict[str, int] = {}
        for phrase in candidates:
            key = phrase.lower()
            frequencies[key] = frequencies.get(key, 0) + 1

        ranked = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        ordered = []
        for key, _ in ranked:
            ordered.append(key)
        return _safe_unique([c for c in candidates if c.lower() in set(dict(ranked).keys())])[:top_k]

    def classify_sentiment(self, text: str) -> Dict[str, Any]:
        if not text.strip():
            return {"Sentiment": "Neutral", "score": 0.0}

        if self.sentiment_pipe:
            result = self.sentiment_pipe(text[:512])[0]
            label = result["label"].upper()
            score = float(result["score"])
            if label == "NEGATIVE":
                sentiment = "Anxious" if score >= 0.55 else "Neutral"
            elif label == "POSITIVE":
                sentiment = "Reassured" if score >= 0.55 else "Neutral"
            else:
                sentiment = "Neutral"
            return {"Sentiment": sentiment, "score": score, "raw_label": label}

        anxious_terms = ["worried", "concern", "anxious", "nervous"]
        reassured_terms = ["relief", "better", "improving", "ok"]
        lower = text.lower()
        if any(t in lower for t in anxious_terms):
            return {"Sentiment": "Anxious", "score": 0.5}
        if any(t in lower for t in reassured_terms):
            return {"Sentiment": "Reassured", "score": 0.5}
        return {"Sentiment": "Neutral", "score": 0.0}

    def classify_intent(self, text: str) -> Dict[str, Any]:
        candidate_labels = [
            "Seeking reassurance",
            "Reporting symptoms",
            "Expressing concern",
            "Providing history",
            "Closing conversation",
        ]
        if self.zero_shot_pipe:
            result = self.zero_shot_pipe(text[:512], candidate_labels=candidate_labels)
            top_label = result["labels"][0]
            return {"Intent": top_label, "scores": dict(zip(result["labels"], result["scores"]))}

        lower = text.lower()
        if any(t in lower for t in ["worried", "hope", "worry", "better soon"]):
            return {"Intent": "Seeking reassurance"}
        if any(t in lower for t in ["pain", "ache", "hurt", "symptom"]):
            return {"Intent": "Reporting symptoms"}
        if "thank" in lower:
            return {"Intent": "Closing conversation"}
        return {"Intent": "Providing history"}

    def structured_summary(self, text: str) -> Dict[str, Any]:
        entities = self.extract_entities(text)
        name = self._infer_patient_name(text) or "Unknown"
        current_status = None
        for sentence in text.split("."):
            if "now" in sentence.lower() or "currently" in sentence.lower():
                current_status = sentence.strip()
                break
        current_status = current_status or "Not clearly stated"
        prognosis = entities["prognosis"][0] if entities["prognosis"] else "Not specified"

        return {
            "Patient_Name": name,
            "Symptoms": entities["symptoms"] or ["Not stated"],
            "Diagnosis": entities["diagnosis"][0] if entities["diagnosis"] else "Not specified",
            "Treatment": entities["treatment"] or ["Not stated"],
            "Current_Status": current_status,
            "Prognosis": prognosis,
        }

    def generate_soap(self, text: str) -> Dict[str, Any]:
        summary = self.structured_summary(text)
        patient_text = " ".join(
            [line.split(":", 1)[1].strip() for line in text.splitlines() if line.lower().startswith("patient:")]
        )
        objective = self._extract_objective(text)

        assessment = {
            "Diagnosis": summary["Diagnosis"],
            "Severity": "Mild, improving" if "improv" in text.lower() else "Not specified",
        }
        plan = {
            "Treatment": ", ".join(summary["Treatment"]),
            "Follow_Up": "Return if pain worsens; continue exercises/physiotherapy as needed.",
        }
        return {
            "Subjective": {
                "Chief_Complaint": ", ".join(summary["Symptoms"]),
                "History_of_Present_Illness": patient_text or "Not captured",
            },
            "Objective": objective,
            "Assessment": assessment,
            "Plan": plan,
        }

    def _extract_objective(self, text: str) -> Dict[str, str]:
        objective = {
            "Physical_Exam": "Not documented",
            "Observations": "Not documented",
        }
        for line in text.splitlines():
            if "Physical exam" in line or "Physical Examination" in line:
                objective["Physical_Exam"] = line.split(":", 1)[-1].strip()
            if "range of motion" in line.lower() or "tenderness" in line.lower():
                objective["Observations"] = line.strip()
        return objective

    def run_all(self, text: str) -> Dict[str, Any]:
        patient_only = " ".join(
            [line.split(":", 1)[1].strip() for line in text.splitlines() if line.lower().startswith("patient:")]
        )
        return {
            "entities": self.extract_entities(text),
            "keywords": self.extract_keywords(text),
            "structured_summary": self.structured_summary(text),
            "sentiment_intent": {
                **self.classify_sentiment(patient_only or text),
                **self.classify_intent(patient_only or text),
            },
            "soap": self.generate_soap(text),
        }


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run medical NLP pipeline locally.")
    parser.add_argument(
        "--file", type=str, default="data/sample_transcript.txt", help="Path to transcript text file."
    )
    parser.add_argument("--output", type=str, default=None, help="Optional path to write JSON output.")
    args = parser.parse_args()

    text = Path(args.file).read_text(encoding="utf-8")
    pipeline = MedicalNLPipeline()
    result = pipeline.run_all(text)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

