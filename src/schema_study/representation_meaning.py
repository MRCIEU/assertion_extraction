"""Semantic reading of the frozen predicate groups.

The records describe the BioRED annotation guideline of 7 April 2022 and the
ChemProt guideline, version 6. They do not change the frozen mappings. A nested
partition is recorded as structural validity. It is not treated as evidence
that the parent name is one biological mechanism.
"""

from __future__ import annotations

REQUIRED_FIELDS = (
    "original_definition",
    "broader_definition",
    "native_categories",
    "distinctions_preserved",
    "distinctions_omitted",
    "additional_meaning",
    "defensible_natural_language",
    "natural_language_note",
    "one_biological_mechanism",
)


def _row(
    definition: str,
    broader: str,
    native: list[str],
    preserved: str,
    omitted: str,
    extra: str,
    *,
    defensible: bool,
    note: str,
    mechanism: bool,
) -> dict:
    return {
        "original_definition": definition,
        "broader_definition": broader,
        "native_categories": native,
        "distinctions_preserved": preserved,
        "distinctions_omitted": omitted,
        "additional_meaning": extra,
        "defensible_natural_language": defensible,
        "natural_language_note": note,
        "one_biological_mechanism": mechanism,
    }


def predicate_interpretations() -> dict:
    """Interpretation of every native label and every broader group."""
    biored_native = {
        "Positive_Correlation": _row(
            "The guideline uses this label when one item is described as raising another, or when their levels move together. The examples include a chemical or a higher dose raising disease risk, a disease raising a measured chemical, a protein or variant contributing to disease, and a chemical raising gene expression.",
            "This is a native category, so the broader representation is the same label.",
            ["Positive_Correlation"],
            "The annotated direction is an increase or a co-movement.",
            "No finer distinction inside this label is represented. Causation, dose response, treatment failure, and a reported correlation share the label.",
            "The word correlation can suggest a statistical coefficient when the text states a causal or experimental effect.",
            defensible=True,
            note="The directional reading matches the guideline examples. It does not identify which kind of increase the abstract states.",
            mechanism=False,
        ),
        "Negative_Correlation": _row(
            "The guideline uses this label when one item lowers another or when a treatment or protective relationship is described. The examples include a drug treating a disease, a disease lowering a chemical, a protein preventing disease, and a chemical lowering gene expression.",
            "This is a native category, so the broader representation is the same label.",
            ["Negative_Correlation"],
            "The annotated direction is a decrease, a treatment, or a protective relationship.",
            "Treatment, prevention, and a reported inverse association are not separated.",
            "A reader can take the name to mean only that two measurements move in opposite directions. The guideline also places therapeutic relationships here.",
            defensible=True,
            note="The label is a directional bin used by the guideline, not a single statistical operation.",
            mechanism=False,
        ),
        "Association": _row(
            "The guideline places a pair here when it cannot be assigned to positive or negative correlation. The examples include an underspecified association, a drug described as safe in a possible disease, a variant only observed in patients, and a chemical modification such as phosphorylation.",
            "This is a native category, so the broader representation is the same label.",
            ["Association"],
            "The pair was annotated and was not given a directional correlation label.",
            "The label combines several reasons for withholding a direction, including missing evidence and modification events.",
            "The name can suggest a specific epidemiological association or a shared mechanism. In the guideline it is the residual bin.",
            defensible=False,
            note="The name is usable as 'annotated, but not a directional correlation'. It does not name one biological relation.",
            mechanism=False,
        ),
        "Bind": _row(
            "The guideline defines binding as a physical interaction: proteins binding one another, a protein binding a promoter, membership of a complex, a chemical binding a promoter, or a protein acting as the receptor of a chemical. If the text says that the binding causes a positive or negative correlation, the guideline says to annotate that correlation instead.",
            "This is a native category, so the broader representation is the same label.",
            ["Bind"],
            "Physical contact is kept apart from a merely directional claim.",
            "The different physical settings are not separated. Binding that is said to cause a correlation is deliberately absent.",
            "The name can be read as every binding event in the text, including the consequential bindings that the guideline moves to a correlation label.",
            defensible=True,
            note="The physical reading is defensible only together with the guideline's exception for consequential binding.",
            mechanism=True,
        ),
        "Comparison": _row(
            "The guideline table assigns Comparison to chemical–chemical pairs. The extracted prose defines Cotreatment and Conversion in adjacent lines and does not give Comparison a separate sentence.",
            "This is a native category, so the broader representation is the same label.",
            ["Comparison"],
            "The pair is a chemical–chemical relation other than the relations with their own prose definitions.",
            "The guideline text recovered here does not say what is being compared or whether one chemical is preferred.",
            "The name can imply a ranked or quantitative comparison that the guideline does not state.",
            defensible=False,
            note="A natural-language reading of 'comparison' is available, but it is not fixed by a recovered prose definition.",
            mechanism=False,
        ),
        "Cotreatment": _row(
            "The guideline defines Cotreatment as combination therapy.",
            "This is a native category, so the broader representation is the same label.",
            ["Cotreatment"],
            "Two chemicals are given together as a treatment.",
            "Synergy, additivity, and the mere fact of co-administration are not separated.",
            "The name can imply that the combination itself was shown to be beneficial, rather than only that the chemicals were used together.",
            defensible=True,
            note="The reading 'combination therapy' follows the guideline sentence.",
            mechanism=False,
        ),
        "Drug_Interaction": _row(
            "The guideline defines a drug interaction as a pharmacodynamic interaction when two chemicals are given together.",
            "This is a native category, so the broader representation is the same label.",
            ["Drug_Interaction"],
            "A pharmacodynamic interaction is distinguished from combination therapy in the guideline headings.",
            "The direction and the clinical consequence of the interaction are not represented.",
            "The name can include pharmacokinetic interactions, which the recovered sentence does not mention.",
            defensible=True,
            note="The pharmacodynamic reading is defensible. Its boundary with Cotreatment is not given a further test in the recovered text.",
            mechanism=False,
        ),
        "Conversion": _row(
            "One guideline heading says that a chemical converts to the other chemical. An earlier chemical–chemical note says to annotate chemical conversion as Association.",
            "This is a native category, so the broader representation is the same label.",
            ["Conversion"],
            "The released label exists separately from Association.",
            "The recovered guideline contains two instructions for a conversion. The mapping cannot say which instruction a given abstract followed.",
            "The name can imply a directed chemical reaction, including cases the other sentence would have left as Association.",
            defensible=False,
            note="The corpus uses the label, but the two guideline sentences were not reconciled here.",
            mechanism=False,
        ),
    }
    biored = {
        "P4": biored_native,
        "P3": {
            "Positive_Correlation": _copy(biored_native["Positive_Correlation"], "The P3 group contains only Positive_Correlation."),
            "Negative_Correlation": _copy(biored_native["Negative_Correlation"], "The P3 group contains only Negative_Correlation."),
            "Association": _copy(biored_native["Association"], "The P3 group contains only Association. The residual character remains."),
            "Bind": _copy(biored_native["Bind"], "The P3 group contains only Bind."),
            "Remaining_Relation": _row(
                "Comparison, Cotreatment, Drug_Interaction, and Conversion keep the definitions above.",
                "Remaining_Relation collects the four chemical–chemical labels that are not correlation, association, or binding.",
                ["Comparison", "Cotreatment", "Drug_Interaction", "Conversion"],
                "These four labels stay separate from correlation, association, and binding.",
                "Comparison, combination therapy, pharmacodynamic interaction, and chemical conversion are no longer distinguished.",
                "The name can suggest that the four labels are leftovers of one chemical relation. They do not share one mechanism.",
                defensible=False,
                note="The name is an honest residual label. It is not a biological generalisation.",
                mechanism=False,
            ),
        },
        "P2": {
            "Positive_Correlation": _copy(biored_native["Positive_Correlation"], "The group is unchanged from P4 through P2."),
            "Negative_Correlation": _copy(biored_native["Negative_Correlation"], "The group is unchanged from P4 through P2."),
            "Other_Relation": _row(
                "The members keep their native definitions: residual association, physical binding, and the four chemical–chemical labels.",
                "Other_Relation collects every positive BioRED label other than the two correlation labels.",
                ["Association", "Bind", "Comparison", "Cotreatment", "Drug_Interaction", "Conversion"],
                "Directional correlation remains outside this group.",
                "Binding, residual association, and every chemical–chemical distinction are omitted.",
                "The name can be read as one other biological relation. It is only the complement of the two correlation labels.",
                defensible=False,
                note="Structural closure does not give this parent a shared mechanism.",
                mechanism=False,
            ),
        },
        "P1": {
            "Any_Relation": _row(
                "Every native positive definition remains available only in the member list.",
                "Any_Relation asserts that some positive annotated relation exists.",
                [
                    "Positive_Correlation",
                    "Negative_Correlation",
                    "Association",
                    "Bind",
                    "Comparison",
                    "Cotreatment",
                    "Drug_Interaction",
                    "Conversion",
                ],
                "The distinction between an annotated relation and the absence of one is preserved.",
                "Direction, binding, association, and every chemical–chemical distinction are omitted.",
                "The schema stores the group as nonpolar because its members do not share a direction. The name can still be read as if those relations were interchangeable findings.",
                defensible=True,
                note="The defensible reading is existential: a positive annotation exists. It does not describe the finding.",
                mechanism=False,
            ),
        },
    }
    drugprot_native = {
        "ACTIVATOR": _row(
            "A chemical that binds a gene product, typically a protein, and increases its activity. The guideline lists stimulator, inducer, potentiator, and enhancer as wording that can express this class.",
            "This is a native category.",
            ["ACTIVATOR"],
            "Direct binding and an increase in activity are both required.",
            "The synonyms are not separate labels.",
            "The name can be applied to an increase that the guideline would call indirect or agonistic.",
            defensible=True,
            note="The direct-binding reading matches the guideline.",
            mechanism=True,
        ),
        "AGONIST": _row(
            "A chemical that binds a receptor and alters the receptor state, producing a biological response. The guideline assigns this general class when the text does not say whether the response increases or decreases.",
            "This is a native category.",
            ["AGONIST"],
            "Receptor agonism with an unstated direction is kept apart from activator and from the two directed agonist labels.",
            "Increase and decrease are both absent by definition of this subclass.",
            "The stored direction is nonpolar. The ordinary use of agonist can still suggest an increase.",
            defensible=True,
            note="The reading is defensible as unspecified-direction agonism, not as activation.",
            mechanism=True,
        ),
        "AGONIST-ACTIVATOR": _row(
            "An agonist that binds a receptor and increases its biological response.",
            "This is a native category.",
            ["AGONIST-ACTIVATOR"],
            "The increase is tied to receptor agonism rather than to the general activator class.",
            "Full and partial agonism are not separated.",
            "Calling it simply an activator would drop the receptor-agonist setting.",
            defensible=True,
            note="The guideline gives this subclass a directional effect. The schema stores the label as nonpolar.",
            mechanism=True,
        ),
        "AGONIST-INHIBITOR": _row(
            "An agonist that binds a receptor and decreases its biological response. The guideline contrasts this inverse-agonist pattern with agonists that increase the response.",
            "This is a native category.",
            ["AGONIST-INHIBITOR"],
            "A decrease produced through agonism stays separate from a plain inhibitor.",
            "The structural details of inverse agonism are not represented.",
            "The word agonist, without the inhibitor suffix, can suggest an increase. The schema direction is nonpolar even though the guideline describes a decrease.",
            defensible=True,
            note="The native label is directional in the guideline. A parent called Agonism does not keep that direction.",
            mechanism=True,
        ),
        "ANTAGONIST": _row(
            "A chemical that reduces the action of another chemical, generally an agonist, commonly by acting at the same receptor.",
            "This is a native category.",
            ["ANTAGONIST"],
            "Blocking another chemical's action is kept apart from directly decreasing the gene product's own activity.",
            "Competitive and non-competitive antagonism are not separated.",
            "The name can be read as a general inhibitor. The guideline does not use it that way.",
            defensible=True,
            note="The receptor-blocking reading is defensible and is narrower than downregulation.",
            mechanism=True,
        ),
        "DIRECT-REGULATOR": _row(
            "A chemical that binds a gene product and changes its activity when the text does not say whether the activity rises or falls.",
            "This is a native category.",
            ["DIRECT-REGULATOR"],
            "Direct binding is represented without a direction.",
            "Increase and decrease are omitted because the annotation did not state them.",
            "Regulator can suggest any influence, including an indirect one. The guideline requires direct binding.",
            defensible=True,
            note="The unspecified-direction reading matches the guideline.",
            mechanism=True,
        ),
        "INDIRECT-UPREGULATOR": _row(
            "A chemical that increases a gene product through other cellular components, or through expression, stability, release, or a post-translational modification, rather than by direct binding.",
            "This is a native category.",
            ["INDIRECT-UPREGULATOR"],
            "The increase and the indirect route are both represented.",
            "Expression, stability, release, and modification are not separate labels.",
            "Upregulator can be read as direct activation.",
            defensible=True,
            note="The indirect reading is defensible and is the reason this label is not placed in DIRECT-REGULATOR.",
            mechanism=False,
        ),
        "INDIRECT-DOWNREGULATOR": _row(
            "A chemical that decreases a gene product through other cellular components, or through expression, stability, release, or a post-translational modification, rather than by direct binding.",
            "This is a native category.",
            ["INDIRECT-DOWNREGULATOR"],
            "The decrease and the indirect route are both represented.",
            "The indirect routes are not separated.",
            "The name can be read as a direct inhibitor.",
            defensible=True,
            note="The indirect reading matches the guideline and stays outside INHIBITOR.",
            mechanism=False,
        ),
        "INHIBITOR": _row(
            "A chemical that binds a gene product and decreases its activity.",
            "This is a native category.",
            ["INHIBITOR"],
            "Direct binding and a decrease are both required.",
            "Reversible, irreversible, and competitive inhibition are not separated.",
            "The name can cover antagonism or an indirect decrease, which have their own labels.",
            defensible=True,
            note="The direct-binding reading matches the guideline.",
            mechanism=True,
        ),
        "PART-OF": _row(
            "A chemical is structurally part of the gene product, for example an amino-acid residue.",
            "This is a native category.",
            ["PART-OF"],
            "Structural inclusion is kept apart from regulation and metabolism.",
            "The kind of structural part is not represented.",
            "Part of can suggest a functional subunit rather than a chemical constituent.",
            defensible=True,
            note="The structural reading matches the guideline example.",
            mechanism=True,
        ),
        "PRODUCT-OF": _row(
            "The chemical is a product of the gene product, including a product of an enzyme or a transporter.",
            "This is a native category.",
            ["PRODUCT-OF"],
            "The chemical is an output rather than a regulator.",
            "Enzymatic product and transported product are not separated.",
            "Product can suggest only a metabolic product, although the guideline also includes transport.",
            defensible=True,
            note="The output reading is defensible if transport is included.",
            mechanism=False,
        ),
        "SUBSTRATE": _row(
            "The gene product acts on the chemical.",
            "This is a native category.",
            ["SUBSTRATE"],
            "The chemical is an input to the gene product's action.",
            "The kind of action is not represented.",
            "Substrate can suggest an enzyme-substrate reaction when the text describes another action on the chemical.",
            defensible=True,
            note="The guideline reading is that the gene product acts on the chemical.",
            mechanism=False,
        ),
        "SUBSTRATE_PRODUCT-OF": _row(
            "The chemical is both acted on by the gene product and a product of it.",
            "This is a native category.",
            ["SUBSTRATE_PRODUCT-OF"],
            "The combined role is not reduced to either role alone.",
            "The reaction or transport step is not represented.",
            "The hyphenated name can be parsed as two unrelated labels rather than as one combined annotation.",
            defensible=True,
            note="The combined reading matches the guideline class.",
            mechanism=False,
        ),
    }
    up_members = ["ACTIVATOR", "INDIRECT-UPREGULATOR"]
    down_members = ["INHIBITOR", "INDIRECT-DOWNREGULATOR"]
    agonism_members = ["AGONIST", "AGONIST-ACTIVATOR", "AGONIST-INHIBITOR"]
    metabolic_members = ["SUBSTRATE", "PRODUCT-OF", "SUBSTRATE_PRODUCT-OF"]
    other_members = [
        "AGONIST",
        "AGONIST-ACTIVATOR",
        "AGONIST-INHIBITOR",
        "ANTAGONIST",
        "DIRECT-REGULATOR",
        "PART-OF",
        "PRODUCT-OF",
        "SUBSTRATE",
        "SUBSTRATE_PRODUCT-OF",
    ]
    any_members = [
        "ACTIVATOR",
        "AGONIST",
        "AGONIST-ACTIVATOR",
        "AGONIST-INHIBITOR",
        "ANTAGONIST",
        "DIRECT-REGULATOR",
        "INDIRECT-DOWNREGULATOR",
        "INDIRECT-UPREGULATOR",
        "INHIBITOR",
        "PART-OF",
        "PRODUCT-OF",
        "SUBSTRATE",
        "SUBSTRATE_PRODUCT-OF",
    ]
    upregulation = _row(
        "ACTIVATOR requires direct binding and an increase. INDIRECT-UPREGULATOR requires an increase without that direct binding.",
        "Upregulation_or_activation groups the two labels that state an increase and are not agonist labels.",
        up_members,
        "An increase is preserved, and agonist labels stay outside the group.",
        "Direct binding is no longer distinguished from an indirect increase.",
        "The word activation is the guideline's term for a direct effect. It can be read onto the indirect member.",
        defensible=False,
        note="A direction-only reading would be narrower than the stored name. The stored name adds a direct-binding implication.",
        mechanism=False,
    )
    downregulation = _row(
        "INHIBITOR requires direct binding and a decrease. INDIRECT-DOWNREGULATOR requires a decrease without that direct binding.",
        "Downregulation_or_inhibition groups those two labels. ANTAGONIST and AGONIST-INHIBITOR stay outside it.",
        down_members,
        "A decrease that the guideline calls inhibitor or indirect downregulator is preserved.",
        "Direct binding is no longer distinguished from an indirect decrease. Antagonism is not included.",
        "Inhibition, in the guideline, is direct. The group name can attach that directness to the indirect member.",
        defensible=False,
        note="The grouping preserves a decrease and drops the route. The name is broader than either native definition.",
        mechanism=False,
    )
    drugprot = {
        "P4": drugprot_native,
        "P3": {
            "Upregulation_or_activation": upregulation,
            "Downregulation_or_inhibition": downregulation,
            "Direct_regulation_unspecified": _copy(
                drugprot_native["DIRECT-REGULATOR"],
                "The P3 group contains only DIRECT-REGULATOR.",
            ),
            "Agonism": _row(
                "AGONIST leaves the direction unstated. AGONIST-ACTIVATOR increases the receptor response. AGONIST-INHIBITOR decreases it.",
                "Agonism groups the three agonist labels.",
                agonism_members,
                "Receptor agonism stays outside ordinary activation, inhibition, and antagonism.",
                "The increase, the decrease, and the unstated direction are omitted.",
                "Agonism can be read as a uniform increase. One member is defined by a decrease.",
                defensible=False,
                note="The parent is a structural grouping of agonist labels, not one directional mechanism.",
                mechanism=False,
            ),
            "Antagonism": _copy(drugprot_native["ANTAGONIST"], "The P3 group contains only ANTAGONIST."),
            "Metabolic_relation": _row(
                "SUBSTRATE, PRODUCT-OF, and SUBSTRATE_PRODUCT-OF keep the input, output, and combined definitions. PRODUCT-OF includes a transporter product.",
                "Metabolic_relation groups those three labels.",
                metabolic_members,
                "The chemical is an input or an output of the gene product, rather than a regulator or a structural part.",
                "Input, output, and the combined role are not distinguished. Transport is not distinguished from metabolism.",
                "Metabolic can imply an enzyme reaction for a relation the guideline would accept as transport.",
                defensible=False,
                note="The group separates these roles from regulation. It does not name one metabolic mechanism.",
                mechanism=False,
            ),
            "Part_of": _copy(drugprot_native["PART-OF"], "The P3 group contains only PART-OF."),
        },
        "P2": {
            "Upregulation_or_activation": upregulation,
            "Downregulation_or_inhibition": downregulation,
            "Other_Relation": _row(
                "The members are the agonist labels, antagonism, unspecified direct regulation, structural inclusion, and the substrate or product labels.",
                "Other_Relation is the complement of the explicit up and down groups.",
                other_members,
                "Explicit upregulation and explicit downregulation stay outside this group.",
                "Agonism, antagonism, direct regulation, structural inclusion, and substrate or product roles are omitted.",
                "The name can be read as one residual biological relation. The members do not share a mechanism.",
                defensible=False,
                note="This is a residual bin. A nested partition does not make it a generalisation.",
                mechanism=False,
            ),
        },
        "P1": {
            "Any_Relation": _row(
                "The thirteen native definitions remain only as members.",
                "Any_Relation asserts that at least one positive DrugProt label applies.",
                any_members,
                "An annotated positive relation is distinguished from a candidate with no positive label.",
                "Direction, route, agonism, antagonism, regulation, structure, and metabolic role are omitted.",
                "The name can suggest that the thirteen findings are the same kind of statement. A candidate may also carry more than one native label.",
                defensible=True,
                note="The defensible reading is existential. It does not identify the finding.",
                mechanism=False,
            ),
        },
    }
    return {"biored": biored, "drugprot": drugprot}


def _copy(row: dict, broader: str) -> dict:
    copied = dict(row)
    copied["broader_definition"] = broader
    return copied


def unresolved_questions() -> list[str]:
    """Questions left open. They are not grounds for editing the frozen mapping."""
    return [
        "The BioRED guideline both defines Conversion and, in another sentence, tells the annotator to mark chemical conversion as Association. The released label is Conversion. Which instruction applies to a particular abstract is unresolved.",
        "The recovered BioRED guideline names Comparison in the chemical–chemical inventory and does not supply a separate prose definition.",
        "The recovered sentences distinguish Cotreatment as combination therapy and Drug_Interaction as a pharmacodynamic interaction, without a further test for abstracts that describe both.",
        "BioRED Association is a residual annotation bin. It is not evidence that its members share a mechanism.",
        "BioRED Bind excludes binding that the text presents as the cause of a correlation. A later source reading cannot treat every binding sentence as this label.",
        "DrugProt Agonism joins an increase, a decrease, and an unstated direction. The parent name can suggest a uniform increase, and the schema stores all three native labels as nonpolar.",
        "Upregulation_or_activation and Downregulation_or_inhibition use activation and inhibition for groups that also contain indirect labels. Those words are narrower in the guideline.",
        "Metabolic_relation includes a product of a transporter. The parent name can imply metabolism.",
        "Remaining_Relation, Other_Relation, and Any_Relation are residual or existential groups. Their structural validity does not supply a biological definition.",
        "No_Relation records that no positive label was annotated. It does not establish that the text contains no relationship.",
    ]


def entity_extension_requirements() -> dict:
    """Methodological requirements for a future UMLS or MeSH entity mapping.

    No ontology resource is read and no mapping is constructed.
    """
    return {
        "current_conditions_retained": ["E0", "E1", "E2"],
        "resources_not_acquired": ["UMLS", "MeSH"],
        "distinctions": {
            "semantic_type": "A UMLS semantic type is a broad category such as Disease or Syndrome. It is not a position in a concept tree.",
            "concept_hierarchy": "A MeSH concept can have several tree numbers, and those paths can have different depths. A single parent is not given by the concept identifier alone.",
            "molecular_form": "A gene, a protein, a transcript, and a variant can share a name. A database class or an identifier pattern does not establish the form mentioned in the sentence.",
            "pharmacological_role": "A pharmacological action is a role of a chemical. It is not an entity type, and it can repeat the relation that the experiment is trying to predict.",
        },
        "future_mapping_must_record": [
            "Every applicable type and every hierarchical path, rather than one silently chosen type.",
            "The depth of each path, so a deeper path is not treated as a larger amount of evidence.",
            "An explicit missing or ambiguous state when the mention has no concept, several concepts, or no agreed path.",
            "The difference between a native corpus category and an ontology category. One must not be renamed into the other without a recorded mapping.",
            "Whether the proposed type is defined by the target relation. An inhibitor type, a substrate type, or a therapeutic role discloses the label and cannot be used as an entity condition.",
        ],
        "present_consequence": "Source passages for the existing entity pairs do not require these finer types. Mention offsets and the native category are sufficient to locate the pair.",
    }
