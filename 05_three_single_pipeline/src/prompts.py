def get_prompt(task: str):
    if task == "finding":
        system = (
            "You are a biomedical information extractor.\n"
            "Your task is to identify all RESULT or FINDING sentences from a given biomedical abstract.\n\n"
            "Definitions and Guidelines:\n"
            "• A 'finding' sentence must report an observed outcome, numerical effect, statistical result (e.g., p-value), or author interpretation.\n"
            "• Sentences describing background, motivation, objectives, or methods are NOT findings.\n"
            "• Use only the original sentence indices; do NOT merge, modify, or split sentences.\n\n"
            "Output strictly one valid JSON object of the form:\n"
            "{\"pmid\": \"<PMID>\", \"finding_ids\": [<int>, ...]}\n"
            "Do NOT include any explanations, formatting, markdown, or extra text."
        )
        fewshot = [
            {"role": "system", "content": system},
            {"role": "user", "content": "PMID: 0000\nSentences:\n[0] We evaluated drug X.\n[1] Drug X reduced viral load by 80%.\nReturn JSON:"},
            {"role": "assistant", "content": '{"pmid": "0000", "finding_ids": [1]}'},
            {"role": "user", "content": "PMID: 9999\nSentences:\n[0] We aimed to assess vitamin C.\n[1] No improvement was observed (p = 0.45).\nReturn JSON:"},
            {"role": "assistant", "content": '{"pmid": "9999", "finding_ids": [1]}'},
        ]
        return system, fewshot

    elif task == "completion":
        system = (
            "You are a biomedical sentence resolver.\n"
            "Your task is to transform biomedical RESULT sentences into standalone factual statements.\n\n"
            "Guidelines:\n"
            "• Each output must contain exactly ONE subject and ONE object.\n"
            "• Resolve all pronouns and coreferences (e.g., replace 'it' with full noun).\n"
            "• Expand all abbreviations (e.g., 'BP' → 'blood pressure').\n"
            "• Preserve all numeric values and modifiers.\n"
            "• If a sentence contains multiple results (e.g., multiple objects/actions), split it into multiple outputs (one per result).\n"
            "• If no result exists, return an empty JSON list [].\n\n"
            "Return ONLY a valid JSON array of the form:\n"
            "[{\"id\": <int>, \"resolved\": \"<sentence>\"}]\n"
            "Each sentence must end with a period."
        )
        fewshot = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Context:\nWe evaluated drug X. Drug X (DX) reduced viral load and blood pressure.\nSentence [1]:\nDX reduced viral load and blood pressure.\nReturn JSON:"},
            {"role": "assistant", "content": '[{"id": 1, "resolved": "Drug X reduced viral load."}, {"id": 1, "resolved": "Drug X reduced blood pressure."}]'},
            {"role": "user", "content": "Context:\nThe treatment (TX) improved outcomes. TX also reduced inflammation.\nSentence [2]:\nTX improved outcomes and reduced inflammation.\nReturn JSON:"},
            {"role": "assistant", "content": '[{"id": 2, "resolved": "The treatment improved outcomes."}, {"id": 2, "resolved": "The treatment reduced inflammation."}]'},
        ]
        return system, fewshot

    elif task == "assertion":
        system = (
            "You are a biomedical assertion extractor.\n"
            "Your task is to extract structured scientific assertions from the given sentence.\n\n"
            "Each assertion must be expressed as a triple: subject, predicate, object.\n\n"
            "Guidelines:\n"
            "• Use standardized biomedical predicates (UMLS-style, e.g., causes, increases, inhibits).\n"
            "• Normalize subjects and objects to canonical biomedical terms.\n"
            "• If no assertion can be made, return: {\"id\": <int>, \"subject\": \"none\", \"predicate\": \"none\", \"object\": \"none\"}."
            "• If applicable, add an optional field \"condition\".\n"
            "• Do not include multiple assertions per JSON.\n"
            "• Output must be a valid JSON syntax ONLY. No explanation or markdown. Never include unmatched brackets or parentheses.\n\n"
            "Format:\n"
            "{\"id\": <int>, \"subject\": \"<subject>\", \"predicate\": \"<predicate>\", \"object\": \"<object>\", \"condition\": \"<optional>\"}"
        )
        fewshot = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Sentence [0]: Drug X reduced blood pressure in hypertensive patients.\nReturn JSON:"},
            {"role": "assistant", "content": '{"id": 0, "subject": "Drug X", "predicate": "reduces", "object": "blood pressure", "condition": "in hypertensive patients"}'},
            {"role": "user", "content": "Sentence [1]: Inhibition of enzyme Y by compound Z leads to apoptosis.\nReturn JSON:"},
            {"role": "assistant", "content": '{"id": 1, "subject": "inhibition of enzyme Y by compound Z", "predicate": "causes", "object": "apoptosis"}'},
            {"role": "user", "content": "Sentence [2]: No valid assertion can be made.\nReturn JSON:"},
            {"role": "assistant", "content": '{"id": 2, "subject": "N/A", "predicate": "N/A", "object": "N/A"}'},
        ]
        return system, fewshot

    else:
        raise ValueError("Unknown task type")