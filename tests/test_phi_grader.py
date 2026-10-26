import ast
import json
import unittest
from pathlib import Path

from deid_eval import phi_grader
from deid_eval.phi_grader import grade, predicted_spans, score_note


def item(source, deid, spans, protected=()):
    phi = []
    for text, category in spans:
        start = source.index(text)
        phi.append({"start": start, "end": start + len(text), "category": category, "text": text})
    return {"source_text": source, "deid_text": deid, "phi_spans": phi, "protected_values": list(protected)}


class GraderTests(unittest.TestCase):
    def test_replacement_becomes_predicted_span(self):
        source = "Seen by Dr. Alvarez today"
        spans = predicted_spans(source, "Seen by Dr. [PROVIDER_NAME] today")
        self.assertEqual([(s["text"], s["label"]) for s in spans], [("Alvarez", "PROVIDER_NAME")])

    def test_placeholder_letters_do_not_align_with_name(self):
        # Without masking, the "M" in "[PATIENT_NAME]" could align with the "M" in "Margaret".
        result = score_note(item("Pt: Margaret Okafor seen", "Pt: [PATIENT_NAME] seen", [("Margaret Okafor", "PATIENT_NAME")]))
        self.assertEqual(result["recall"], 1.0)

    def test_partial_redaction_counts_as_leak(self):
        source = "Lives at 12 Birch Lane."
        result = score_note(item(source, "Lives at 12 [ADDRESS].", [("12 Birch Lane", "ADDRESS")]))
        self.assertEqual(result["fn"], 1)
        self.assertTrue(result["residual_phi"])

    def test_verbatim_repeat_counts_as_leak(self):
        source = "Ms. Ruiz called. Ms. Ruiz agrees."
        second = source.rindex("Ruiz")
        record = {
            "source_text": source,
            "deid_text": "Ms. [PATIENT_NAME] called. Ms. Ruiz agrees.",
            "phi_spans": [
                {"start": 4, "end": 8, "category": "PATIENT_NAME", "text": "Ruiz"},
                {"start": second, "end": second + 4, "category": "PATIENT_NAME", "text": "Ruiz"},
            ],
            "protected_values": [],
        }
        result = score_note(record)
        # The first mention was replaced, but its text still appears in the output.
        self.assertEqual(result["tp"], 0)
        self.assertEqual(result["fn"], 2)

    def test_acuity_over_redaction_is_false_positive(self):
        source = "Seen 03/14/2026. VA OD 20/40."
        record = item(source, "Seen [DATE]. VA OD [DATE].", [("03/14/2026", "DATE")], ["20/40", "OD"])
        result = score_note(record)
        self.assertEqual(result["fp"], 1)
        self.assertLess(result["precision"], 1.0)
        self.assertEqual(result["by_category"]["DATE"]["fp"], 1)
        self.assertEqual(result["clinical_preservation"], 0.5)

    def test_clean_note(self):
        source = "Seen 03/14/2026 by Dr. Alvarez. VA OD 20/40, IOP 12/11 mmHg."
        record = item(
            source,
            "Seen [DATE] by Dr. [PROVIDER_NAME]. VA OD 20/40, IOP 12/11 mmHg.",
            [("03/14/2026", "DATE"), ("Alvarez", "PROVIDER_NAME")],
            ["20/40", "OD", "12/11"],
        )
        result = score_note(record)
        self.assertFalse(result["residual_phi"])
        self.assertEqual(result["clinical_preservation"], 1.0)
        self.assertEqual((result["recall"], result["precision"]), (1.0, 1.0))
        self.assertEqual(grade({}, record), 1.0)

    def test_token_bounded_match(self):
        self.assertEqual(phi_grader.occurrences("20/40", "VA 20/400"), [])
        self.assertEqual(phi_grader.occurrences("12/11", "on 12/11/2025"), [])
        self.assertEqual(len(phi_grader.occurrences("Ruiz", "Ms. Ruiz's chart")), 1)

    def test_whitespace_only_edits_are_ignored(self):
        source = "Line one.\n\nLine two."
        self.assertEqual(predicted_spans(source, "Line one.\nLine two."), [])

    def test_malformed_item_returns_zero(self):
        record = item("Seen by Dr. Alvarez", "Seen by Dr. [PROVIDER_NAME]", [("Alvarez", "PROVIDER_NAME")])
        del record["deid_text"]
        self.assertEqual(grade({}, record), 0.0)
        self.assertEqual(grade({}, None), 0.0)
        self.assertEqual(grade({}, {"source_text": "x", "deid_text": "x", "phi_spans": "not json"}), 0.0)
        bad_offsets = item("Seen by Dr. Alvarez", "Seen", [("Alvarez", "PROVIDER_NAME")])
        bad_offsets["phi_spans"][0]["start"] = 0
        self.assertEqual(grade({}, bad_offsets), 0.0)

    def test_empty_output_scores_zero(self):
        record = item("Seen by Dr. Alvarez", "", [("Alvarez", "PROVIDER_NAME")])
        self.assertEqual(grade({}, record), 0.0)
        self.assertTrue(score_note(record)["residual_phi"])

    def test_grade_accepts_json_strings_and_no_spans(self):
        record = item("Seen by Dr. Alvarez", "Seen by Dr. [PROVIDER_NAME]", [("Alvarez", "PROVIDER_NAME")], ["Dr."])
        record["phi_spans"] = json.dumps(record["phi_spans"])
        record["protected_values"] = json.dumps(record["protected_values"])
        self.assertEqual(grade({}, record), 1.0)
        self.assertEqual(grade({}, item("VA 20/40", "VA 20/40", [])), 1.0)

    def test_partial_recall_score(self):
        source = "Dr. Alvarez saw Ana Ruiz."
        record = item(source, "Dr. [PROVIDER_NAME] saw Ana Ruiz.", [("Alvarez", "PROVIDER_NAME"), ("Ana Ruiz", "PATIENT_NAME")])
        self.assertEqual(grade({}, record), 0.5)

    def test_precision_entry_point(self):
        source = "Seen by Dr. Alvarez. VA OD 20/40."
        record = item(source, "Seen by Dr. [PROVIDER_NAME]. VA OD [DATE].", [("Alvarez", "PROVIDER_NAME")], ["20/40", "OD"])
        self.assertEqual(phi_grader.grade_precision({}, record), 0.5)
        self.assertEqual(grade({}, record), 1.0)
        clean = item("VA OD 20/40.", "VA OD 20/40.", [], ["20/40"])
        self.assertEqual(phi_grader.grade_precision({}, clean), 1.0)

    def test_preservation_entry_point(self):
        record = item("VA OD 20/40, IOP 12/11 mmHg.", "VA OD 20/40, IOP [DATE] mmHg.", [], ["20/40", "12/11"])
        self.assertEqual(phi_grader.grade_preservation({}, record), 0.5)
        self.assertEqual(phi_grader.grade_preservation({}, item("No values.", "No values.", [])), 1.0)

    def test_all_entry_points_fail_closed(self):
        record = item("Seen by Dr. Alvarez", "Seen by Dr. [PROVIDER_NAME]", [("Alvarez", "PROVIDER_NAME")], ["Dr."])
        missing = {k: v for k, v in record.items() if k != "deid_text"}
        empty = dict(record, deid_text="  ")
        for entry in (grade, phi_grader.grade_precision, phi_grader.grade_preservation):
            self.assertEqual(entry({}, missing), 0.0, entry.__name__)
            self.assertEqual(entry({}, empty), 0.0, entry.__name__)
            self.assertEqual(entry({}, None), 0.0, entry.__name__)
            self.assertEqual(entry({}, record), 1.0, entry.__name__)

    def test_leak_reasons(self):
        source = "Dr. Alvarez saw Ms. Ruiz at 12 Birch Lane. Ms. Ruiz agreed."
        first, second = source.index("Ruiz"), source.rindex("Ruiz")
        record = item(
            source,
            "Dr. Alvarez saw Ms. [PATIENT_NAME] at 12 [ADDRESS]. Ms. Ruiz agreed.",
            [("Alvarez", "PROVIDER_NAME"), ("12 Birch Lane", "ADDRESS")],
        )
        for start in (first, second):
            record["phi_spans"].append({"start": start, "end": start + 4, "category": "PATIENT_NAME", "text": "Ruiz"})
        reasons = {(span["text"], span["start"]): span["reason"] for span in score_note(record)["leaked"]}
        self.assertEqual(reasons, {
            ("Alvarez", 4): "missed",
            ("12 Birch Lane", source.index("12")): "partial",
            # The first mention was replaced, but the same name survives in the second mention.
            ("Ruiz", first): "repeat",
            ("Ruiz", second): "missed",
        })
        failed = score_note(dict(record, deid_text=""))
        self.assertEqual({span["reason"] for span in failed["leaked"]}, {"failed"})

    def test_detected_spans_have_no_reason_entry(self):
        record = item("Seen by Dr. Alvarez", "Seen by Dr. [PROVIDER_NAME]", [("Alvarez", "PROVIDER_NAME")])
        self.assertEqual(score_note(record)["leaked"], [])

    def test_lowercase_placeholders_from_other_tools(self):
        source = "Pt: Margaret Okafor  DOB 03/14/1951  Seen by Dr. Henry Lindqvist."
        record = item(source, "Pt: [name]  DOB [date]  Seen by Dr. [name].",
                      [("Margaret Okafor", "PATIENT_NAME"), ("03/14/1951", "DATE"), ("Henry Lindqvist", "PROVIDER_NAME")])
        self.assertEqual(score_note(record)["recall"], 1.0)
        over = item("VA OD 20/40.", "VA OD [date].", [], ["20/40"])
        self.assertEqual(score_note(over)["by_category"]["DATE"]["fp"], 1)

    def test_punctuation_left_around_placeholders_is_not_a_leak(self):
        source = "Call (503) 555-0142 or write to 12 Birch Lane, Fairview, OR 97035."
        record = item(source, "Call ([PHONE]) [PHONE] or write to [ADDRESS], [ADDRESS], [ADDRESS] [ADDRESS].",
                      [("(503) 555-0142", "PHONE"), ("12 Birch Lane, Fairview, OR 97035", "ADDRESS")])
        self.assertEqual(score_note(record)["recall"], 1.0)
        digits_left = item(source, "Call (503) [PHONE] or write to [ADDRESS].",
                           [("(503) 555-0142", "PHONE"), ("12 Birch Lane, Fairview, OR 97035", "ADDRESS")])
        self.assertEqual([l["reason"] for l in score_note(digits_left)["leaked"]], ["partial"])

    def test_align_uses_output_coordinates(self):
        source = "Pt: Ana Ruiz, VA 20/40."
        deid = "Pt: [PATIENT_NAME], VA [DATE]."
        ops = phi_grader.align(source, deid)
        replaced = [(source[i1:i2], deid[j1:j2]) for tag, i1, i2, j1, j2 in ops if tag != "equal"]
        self.assertEqual(replaced, [("Ana Ruiz", "[PATIENT_NAME]"), ("20/40", "[DATE]")])
        # Equal opcodes cover identical text in both strings, and the ranges tile both strings.
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                self.assertEqual(source[i1:i2], deid[j1:j2])
        self.assertEqual((ops[0][1], ops[0][3], ops[-1][2], ops[-1][4]), (0, 0, len(source), len(deid)))

    def test_module_is_standalone_and_under_256kb(self):
        source = Path(phi_grader.__file__).read_text(encoding="utf-8")
        self.assertLess(len(source.encode("utf-8")), 256 * 1024)
        # The Foundry grader sandbox statically rejects calls with these names, even re.compile.
        forbidden = {"compile", "exec", "eval", "open", "__import__"}
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                self.assertNotIn(name, forbidden, "line %d" % node.lineno)
        for line in source.splitlines():
            if line.startswith(("import ", "from ")):
                self.assertRegex(line, r"^(from __future__|import (json|re)$|from difflib )")
        namespace = {}
        exec(compile(source, "phi_grader_foundry", "exec"), namespace)
        record = item("Seen by Dr. Alvarez", "Seen by Dr. [PROVIDER_NAME]", [("Alvarez", "PROVIDER_NAME")])
        self.assertEqual(namespace["grade"]({}, record), grade({}, record))


if __name__ == "__main__":
    unittest.main()
