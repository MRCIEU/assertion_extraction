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
            {
                "role": "system", 
                "content": system
            },
            {
                "role": "user", 
                "content": "PMID: 0000\nSentences:\n[0] We evaluated drug X.\n[1] Drug X reduced viral load by 80%.\nReturn JSON:"},
            {
                "role": "assistant", 
                "content": '{"pmid": "0000", "finding_ids": [1]}'
            },
            {
                "role": "user", 
                "content": "PMID: 9999\nSentences:\n[0] We aimed to assess vitamin C.\n[1] No improvement was observed (p = 0.45).\nReturn JSON:"
            },
            {
                "role": "assistant", 
                "content": '{"pmid": "9999", "finding_ids": [1]}'
            },
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
            "• Preserve all numeric values, units, and qualifiers (e.g., '80%', 'p < 0.05').\n"
            "• Do not rephrase or add interpretation — keep wording close to the original meaning.\n"
            "• If a sentence contains multiple results (e.g., multiple objects/actions), split it into multiple outputs (one per result).\n"
            "• If no resolvable result exists, return an empty array: []\n\n"
            "Output Rules:\n"
            "• Always return a pure JSON **array** of resolved sentences.\n"
            "• Never return a raw object. Wrap even a single result in square brackets: [ ... ]\n"
            "• Do NOT return an object with a 'result' field (e.g., {\"result\": [...]}) — that is INVALID.\n"
            "• Each item in the array must be of the format: {\"id\": <int>, \"resolved\": \"<sentence>\"}\n"
            "• Each sentence must end with a period.\n"
            "• Do not include explanation, comments, or markdown — return JSON **only**."
        )
        fewshot = [
            {
                "role": "system", 
                "content": system
            },
            {
                "role": "user", 
                "content": "Context:\nWe evaluated drug X. Drug X (DX) reduced viral load and blood pressure.\nSentence [1]:\nDX reduced viral load and blood pressure.\nReturn JSON:"
            },
            {
                "role": "assistant", 
                "content": '[{"id": 1, "resolved": "Drug X reduced viral load."}, {"id": 1, "resolved": "Drug X reduced blood pressure."}]'
            },
            {
                "role": "user", 
                "content": "Context:\nThe treatment (TX) improved outcomes. TX also reduced inflammation.\nSentence [2]:\nTX improved outcomes and reduced inflammation.\nReturn JSON:"
            },
            {
                "role": "assistant", 
                "content": '[{"id": 2, "resolved": "The treatment improved outcomes."}, {"id": 2, "resolved": "The treatment reduced inflammation."}]'
            },
            {
                "role": "user", 
                "content": "Context:\nThis study was designed to test a hypothesis.\nSentence [3]:\nThis study was designed to test a hypothesis.\nReturn JSON:"
            },
            {
                "role": "assistant", 
                "content": '[]'
            },
        ]
        return system, fewshot

    elif task == "assertion":
        system = (
            "You are a biomedical assertion extractor.\n"
            "Your task is to extract **exactly one structured scientific assertion** from the given sentence.\n\n"
    
            "Each assertion must be expressed as a triple: subject, predicate, object.\n"
            "If applicable, also include an optional field: condition.\n\n"
    
            "Strict rules:\n"
            "• Use standardized biomedical predicates (e.g., causes, increases, inhibits).\n"
            "• Normalize subjects and objects to canonical biomedical terms (e.g., 'high BP' → 'blood pressure').\n"
            "• Do NOT include more than one assertion.\n"
            "• Do NOT return a list.\n"
            "• Do NOT wrap inside 'result'.\n"
            "• Do NOT include any markdown or explanation.\n\n"
    
            "Format:\n"
            "{\"id\": <int>, \"subject\": \"<subject>\", \"predicate\": \"<predicate>\", \"object\": \"<object>\", \"condition\": \"<optional>\"}\n"
            "If no assertion can be extracted, return exactly:\n"
            "{\"id\": <int>, \"subject\": \"N/A\", \"predicate\": \"N/A\", \"object\": \"N/A\"}"
        )
    
        fewshot = [
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": "Sentence [0]: Drug X reduced blood pressure in hypertensive patients.\nReturn JSON:"
            },
            {
                "role": "assistant",
                "content": '{"id": 0, "subject": "Drug X", "predicate": "reduces", "object": "blood pressure", "condition": "in hypertensive patients"}'
            },
            {
                "role": "user",
                "content": "Sentence [1]: Inhibition of enzyme Y by compound Z leads to apoptosis.\nReturn JSON:"
            },
            {
                "role": "assistant",
                "content": '{"id": 1, "subject": "inhibition of enzyme Y by compound Z", "predicate": "causes", "object": "apoptosis"}'
            },
            {
                "role": "user",
                "content": "Sentence [2]: No valid assertion can be made.\nReturn JSON:"
            },
            {
                "role": "assistant",
                "content": '{"id": 2, "subject": "N/A", "predicate": "N/A", "object": "N/A"}'
            }
        ]
        return system, fewshot

    else:
        raise ValueError("Unknown task type")