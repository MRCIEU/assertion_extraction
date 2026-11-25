"""Biological propositions for relation labels.

Version 2 described how a dataset annotates a pair. Version 3 states the
biological relationship, or states that the guideline does not define one.
A reporting sentence says that the source reports the proposition. It does
not say that the dataset annotated it, and it does not treat discussion as
supporting evidence.
"""

from __future__ import annotations

VERBALISATION_VERSION = "representation_verbalisation_v4"
PREVIOUS_VERBALISATION_VERSION = "representation_verbalisation_v2"

# Each entry is (proposition body, definition_status).
# BioRED relations are not directional. DrugProt keeps chemical then gene roles.
_PROPOSITIONS = {
    "Positive_Correlation": (
        "{a} and {b} are positively associated: their levels or activities move together, or one is described as increasing the other. No specific mechanism is stated.",
        "defined",
    ),
    "Negative_Correlation": (
        "{a} and {b} are negatively associated: their levels or activities move in opposite directions, or one is described as decreasing or countering the other. No specific mechanism is stated.",
        "defined",
    ),
    "Association": (
        "{a} and {b} are associated. The relation has no stated direction and no more specific relation from the BioRED inventory.",
        "defined",
    ),
    "Bind": (
        "{a} and {b} bind one another.",
        "defined",
    ),
    "Comparison": (
        "A comparison is reported between {a} and {b}.",
        "unresolved_definition",
    ),
    "Cotreatment": (
        "{a} and {b} are administered together.",
        "defined",
    ),
    "Drug_Interaction": (
        "{a} and {b} have a pharmacodynamic interaction when they are administered together.",
        "defined",
    ),
    "Conversion": (
        "A chemical conversion relates {a} and {b}. The relation is not directional.",
        "unresolved_overlap",
    ),
    "RELATED_TO": (
        "{a} and {b} stand in a positive relation from the task inventory. The specific relation is not stated.",
        "defined",
    ),
    "Any_Relation": (
        "{a} and {b} stand in a positive relation from the task inventory. The specific relation is not stated.",
        "defined",
    ),
    "ACTIVATOR": (
        "The chemical {a} increases the activity or amount of the gene or gene product {b}.",
        "defined",
    ),
    "AGONIST": (
        "The chemical {a} acts as an agonist of the gene or gene product {b}.",
        "unresolved_direction",
    ),
    "AGONIST-ACTIVATOR": (
        "The chemical {a} is an agonist that increases the activity or amount of the gene or gene product {b}.",
        "defined",
    ),
    "AGONIST-INHIBITOR": (
        "The chemical {a} is an agonist that decreases the activity or amount of the gene or gene product {b}.",
        "defined",
    ),
    "ANTAGONIST": (
        "The chemical {a} acts as an antagonist of the gene or gene product {b}.",
        "defined",
    ),
    "DIRECT-REGULATOR": (
        "The chemical {a} directly regulates the gene or gene product {b}. The direction is not stated.",
        "defined",
    ),
    "INDIRECT-DOWNREGULATOR": (
        "The chemical {a} indirectly decreases the activity or amount of the gene or gene product {b}.",
        "defined",
    ),
    "INDIRECT-UPREGULATOR": (
        "The chemical {a} indirectly increases the activity or amount of the gene or gene product {b}.",
        "defined",
    ),
    "INHIBITOR": (
        "The chemical {a} decreases the activity or amount of the gene or gene product {b}.",
        "defined",
    ),
    "PART-OF": (
        "The chemical {a} is part of the gene or gene product {b}.",
        "defined",
    ),
    "PRODUCT-OF": (
        "The chemical {a} is a product of the gene or gene product {b}.",
        "defined",
    ),
    "SUBSTRATE": (
        "The chemical {a} is a substrate of the gene or gene product {b}.",
        "defined",
    ),
    "SUBSTRATE_PRODUCT-OF": (
        "The chemical {a} is both a substrate and a product of the gene or gene product {b}.",
        "defined",
    ),
}

_REJECTED = {"missing", "invalid", "rejected"}


def prompt_label(label: str) -> str:
    """Historical prompt strings. Renaming RELATED_TO does not authorise a new generation."""
    if label == "RELATED_TO":
        return "Any_Relation"
    if label == "NO_RELATION":
        return "No_Relation"
    return label


def verbalise(
    label: str | None,
    argument_1: str,
    argument_2: str,
    *,
    members: list[str] | None = None,
    condition: str = "proposition",
    corpus: str = "biored",
    prediction_status: str = "available",
) -> dict:
    """One versioned proposition, or a status when no positive assertion exists."""
    if condition not in {"proposition", "reporting"}:
        raise ValueError(condition)
    if prediction_status in _REJECTED:
        return _empty(label, condition, prediction_status)
    if not label:
        return _empty(label, condition, "not_produced")
    if label in {"No_Relation", "NO_RELATION"}:
        return _empty(label, condition, "no_positive_assertion")
    if not str(argument_1 or "").strip() or not str(argument_2 or "").strip():
        return _empty(label, condition, "unresolved_entity")
    native_members = list(members or [label])
    if len(native_members) == 1:
        member = native_members[0]
        template = _PROPOSITIONS.get(member) or _PROPOSITIONS.get(label)
        if template is None:
            return _empty(label, condition, "unresolved_template")
        text, definition = template
        text = text.format(a=argument_1, b=argument_2)
        status = "verbalised"
    else:
        pieces = []
        for member in native_members:
            template = _PROPOSITIONS.get(member)
            if template is None:
                return _empty(label, condition, "unresolved_template")
            pieces.append(template[0].format(a=argument_1, b=argument_2))
        numbered = " ".join(f"({index}) {piece}" for index, piece in enumerate(pieces, start=1))
        text = f"One of the following holds, and they are not one mechanism: {numbered}"
        definition = "disjunction"
        status = "verbalised"
    if condition == "reporting":
        text = "The source reports that " + text[0].lower() + text[1:]
    return {
        "version": VERBALISATION_VERSION,
        "status": status,
        "label": label,
        "condition": condition,
        "text": text,
        "members": native_members,
        "definition_status": definition,
        "adds_causation": False,
        "adds_certainty": False,
        "discussion_is_not_support": True,
        "residual_is_a_disjunction": len(native_members) > 1,
        "corpus_roles": "chemical_then_gene" if corpus == "drugprot" else "non_directional",
    }


def _empty(label, condition: str, status: str) -> dict:
    return {
        "version": VERBALISATION_VERSION,
        "status": status,
        "label": label,
        "condition": condition,
        "text": None,
        "adds_causation": False,
        "adds_certainty": False,
        "discussion_is_not_support": True,
        "residual_is_a_disjunction": False,
    }
