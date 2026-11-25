"""Fixed verbalisation of a relation label.

Templates follow the annotation definitions. They do not add causation,
mechanism, or certainty that the label does not contain. A parent with
several native members is spoken as that set.
"""

from __future__ import annotations

VERBALISATION_VERSION = "representation_verbalisation_v2"

_NATIVE = {
    "Positive_Correlation": "are annotated with a positive correlation, including an increase or a co-movement in the BioRED guideline",
    "Negative_Correlation": "are annotated with a negative correlation, including a decrease, treatment, or protective relationship in the BioRED guideline",
    "Association": "are annotated with Association, the BioRED residual category for pairs not assigned a directional correlation",
    "Bind": "are annotated with a physical binding relation, subject to the guideline exception that a stated consequential correlation is annotated as that correlation",
    "Comparison": "are annotated with Comparison; the recovered guideline does not give this label a separate prose definition",
    "Cotreatment": "are annotated as a combination treatment",
    "Drug_Interaction": "are annotated with a pharmacodynamic interaction between chemicals given together",
    "Conversion": "are annotated with a chemical conversion; an overlapping Association instruction remains unresolved",
    "ACTIVATOR": "are annotated with direct upregulation or activation of the gene or gene product by the chemical",
    "AGONIST": "are annotated with an agonist relation; the guideline direction of the effect is not uniform",
    "AGONIST-ACTIVATOR": "are annotated with an agonist that activates the gene or gene product",
    "AGONIST-INHIBITOR": "are annotated with an agonist that inhibits the gene or gene product",
    "ANTAGONIST": "are annotated with an antagonist relation",
    "DIRECT-REGULATOR": "are annotated with direct regulation whose direction is not specified",
    "INDIRECT-DOWNREGULATOR": "are annotated with indirect downregulation",
    "INDIRECT-UPREGULATOR": "are annotated with indirect upregulation",
    "INHIBITOR": "are annotated with direct inhibition or downregulation",
    "PART-OF": "are annotated with the chemical as part of the gene or gene product",
    "PRODUCT-OF": "are annotated with the chemical as a product of the gene or gene product",
    "SUBSTRATE": "are annotated with the chemical as a substrate of the gene or gene product",
    "SUBSTRATE_PRODUCT-OF": "are annotated with the chemical as both substrate and product",
}


def verbalise(
    label: str | None,
    argument_1: str,
    argument_2: str,
    *,
    members: list[str] | None = None,
    condition: str = "proposition",
    corpus: str = "biored",
) -> dict:
    """Return one versioned sentence, or a status when no assertion was produced."""
    if condition not in {"proposition", "reporting"}:
        raise ValueError(condition)
    if not label:
        return {
            "version": VERBALISATION_VERSION,
            "status": "not_produced",
            "text": None,
            "condition": condition,
        }
    if label in {"No_Relation", "NO_RELATION"}:
        text = (
            f"No positive relation between {argument_1} and {argument_2} is annotated "
            "within the dataset scope. This does not establish that the entities have no biological relationship."
        )
        return _done(label, condition, text, members or [], causal=False)
    native_members = list(members or [label])
    if len(native_members) > 1:
        listed = ", ".join(native_members)
        body = f"are annotated with one of these permitted categories, which are not one shared mechanism: {listed}"
    else:
        body = _NATIVE.get(label)
        if body is None:
            return {
                "version": VERBALISATION_VERSION,
                "status": "unresolved_template",
                "text": None,
                "label": label,
                "condition": condition,
            }
    if corpus == "drugprot":
        pair = f"The chemical {argument_1} and the gene or gene product {argument_2} {body}."
    else:
        pair = f"{argument_1} and {argument_2} {body}."
    if condition == "reporting":
        pair = "The source annotation records that " + pair[0].lower() + pair[1:]
    return _done(label, condition, pair, native_members, causal=False)


def _done(label: str, condition: str, text: str, members: list[str], *, causal: bool) -> dict:
    return {
        "version": VERBALISATION_VERSION,
        "status": "verbalised",
        "label": label,
        "condition": condition,
        "text": text,
        "members": members,
        "adds_causation": causal,
        "adds_certainty": False,
        "residual_is_a_disjunction": len(members) > 1,
    }
