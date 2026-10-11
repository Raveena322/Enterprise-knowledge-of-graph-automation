"""
Generates a synthetic-but-realistic dataset of (sentence, subject, relation, object, label)
candidate knowledge-graph triples drawn from enterprise-style text.
label = 1 if the (subject, relation, object) triple is supported by the sentence, else 0.
"""
import random

PEOPLE = ["Alvarez", "Chen", "Osei", "Kapoor", "Nguyen", "Rossi", "Fatima", "Bianchi",
          "Okonkwo", "Larsen", "Suzuki", "Petrov", "Diallo", "Moreau", "Silva", "Ivanov",
          "Haddad", "Kowalski", "Reyes", "Tanaka"]
DEPTS = ["Finance", "Engineering", "Marketing", "Human Resources", "Legal",
         "Procurement", "Sales", "Data Science", "IT Operations", "Customer Support"]
PRODUCTS = ["Falcon CRM", "Nimbus ERP", "Vertex Analytics", "Orion Billing",
            "Atlas Ticketing", "Zenith Payroll", "Quartz Reporting", "Helix Inventory"]
PROJECTS = ["Project Meridian", "Project Solstice", "Project Cobalt", "Project Halcyon",
            "Project Ember", "Project Tundra"]

REL_TEMPLATES = {
    "works_for": [
        "{s} is a member of the {o} department.",
        "{s} currently works in {o}.",
        "As of this quarter, {s} has been assigned to the {o} team.",
        "{s} joined the {o} division last year.",
    ],
    "reports_to": [
        "{s} reports to {o}.",
        "{o} is the direct manager of {s}.",
        "According to the org chart, {s}'s supervisor is {o}.",
        "{s} was reassigned to report to {o} after the reorg.",
    ],
    "manages": [
        "{s} manages the {o} team.",
        "{o} is overseen by {s} on a day-to-day basis.",
        "{s} is responsible for managing {o}.",
    ],
    "uses": [
        "The {s} department relies on {o} for daily operations.",
        "{o} is the primary tool used by {s}.",
        "{s} adopted {o} to replace its legacy system.",
    ],
    "produces": [
        "{s} is the product team behind {o}.",
        "{o} was developed and shipped by {s}.",
        "{s} owns the roadmap for {o}.",
    ],
    "contributes_to": [
        "{s} is contributing engineering resources to {o}.",
        "{s} has been assigned to work on {o} this cycle.",
        "{o} depends on deliverables from {s}.",
    ],
}

RELATION_DOMAINS = {
    "works_for": (PEOPLE, DEPTS),
    "reports_to": (PEOPLE, PEOPLE),
    "manages": (PEOPLE, DEPTS),
    "uses": (DEPTS, PRODUCTS),
    "produces": (DEPTS, PRODUCTS),
    "contributes_to": (DEPTS, PROJECTS),
}

RELATIONS = list(REL_TEMPLATES.keys())


def _typo(text, rate=0.0):
    """Character-level noise to emulate noisy/OCR-extracted text."""
    if rate <= 0:
        return text
    chars = list(text)
    n_edits = max(1, int(len(chars) * rate))
    for _ in range(n_edits):
        idx = random.randrange(len(chars))
        op = random.choice(["drop", "swap", "dup"])
        if op == "drop" and len(chars) > 1:
            del chars[idx]
        elif op == "swap" and idx < len(chars) - 1:
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif op == "dup":
            chars.insert(idx, chars[idx])
    return "".join(chars)


def _make_positive(relation):
    subj_pool, obj_pool = RELATION_DOMAINS[relation]
    s, o = random.choice(subj_pool), random.choice(obj_pool)
    template = random.choice(REL_TEMPLATES[relation])
    sent = template.format(s=s, o=o)
    return sent, s, relation, o, 1


def _make_negative(relation):
    """Hard negatives: (a) right relation, wrong pairing; (b) sentence about one
    relation but candidate labelled with another; (c) unrelated sentence."""
    subj_pool, obj_pool = RELATION_DOMAINS[relation]
    kind = random.choice(["mismatch", "wrong_relation", "unrelated_sentence"])
    if kind == "mismatch":
        s1, o1 = random.choice(subj_pool), random.choice(obj_pool)
        template = random.choice(REL_TEMPLATES[relation])
        sent = template.format(s=s1, o=o1)
        o2 = random.choice([x for x in obj_pool if x != o1])
        return sent, s1, relation, o2, 0
    elif kind == "wrong_relation":
        s1, o1 = random.choice(subj_pool), random.choice(obj_pool)
        template = random.choice(REL_TEMPLATES[relation])
        sent = template.format(s=s1, o=o1)
        other_rel = random.choice([r for r in RELATIONS if r != relation])
        return sent, s1, other_rel, o1, 0
    else:
        other_rel = random.choice([r for r in RELATIONS if r != relation])
        subj_pool2, obj_pool2 = RELATION_DOMAINS[other_rel]
        s2, o2 = random.choice(subj_pool2), random.choice(obj_pool2)
        template = random.choice(REL_TEMPLATES[other_rel])
        sent = template.format(s=s2, o=o2)
        subj_pool, obj_pool = RELATION_DOMAINS[relation]
        s1, o1 = random.choice(subj_pool), random.choice(obj_pool)
        return sent, s1, relation, o1, 0


def build_dataset(n_per_relation=220, noise_rate=0.0):
    rows = []
    for relation in RELATIONS:
        for _ in range(n_per_relation):
            if random.random() < 0.5:
                sent, s, r, o, y = _make_positive(relation)
            else:
                sent, s, r, o, y = _make_negative(relation)
            if noise_rate > 0:
                sent = _typo(sent, noise_rate)
            rows.append({"sentence": sent, "subject": s, "relation": r, "object": o, "label": y})
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    random.seed(42)
    data = build_dataset()
    pos = sum(r["label"] for r in data)
    print(f"Total examples: {len(data)}, positive: {pos}, negative: {len(data) - pos}")
