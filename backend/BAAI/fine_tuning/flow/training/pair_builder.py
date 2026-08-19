"""
pair_builder.py

Generates (anchor, positive, negative) TRIPLETS for triplet-loss
fine-tuning of BAAI/bge-base-en-v1.5:
    anchor   = project/task text
    positive = text of an employee who scores well against the task
    negative = text of an employee who scores poorly against the task

Replaces the old binary-labeled-pair approach. Output:
    datasets/train_pairs.csv   (80%)
    datasets/test_pairs.csv    (20%)
"""

import random
import ast

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# -----------------------------
# Config (inlined -- no external config.py in this project)
# -----------------------------

EMPLOYEES_FILE = "datasets/employee_dataset.csv"
PROJECTS_FILE = "datasets/project_dataset.csv"

TRAIN_PAIRS_FILE = "datasets/train_pairs.csv"
TEST_PAIRS_FILE = "datasets/test_pairs.csv"

TOP_POSITIVES = 10          # top-N scoring employees per project -> positives
NEGATIVES_PER_POSITIVE = 3  # how many negatives to pair with each positive
TEST_SIZE = 0.2
RANDOM_SEED = 42

TASK_SKILL_COLS = ["languages", "technologies", "tools"]
EMPLOYEE_SKILL_COLS = ["languages", "technologies", "tools", "other_skills"]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -----------------------------
# Helpers
# -----------------------------

def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_list_field(value) -> list:
    """
    Employee skill/rating fields are Python-list-literal strings, e.g.
    "['Python', 'CUDA']" or "[2, 4]". Project skill fields are plain
    comma-separated text, e.g. "C#, C++, ShaderLab". This handles both
    -- splitting the bracketed string on "," directly (as the original
    triplet_generator.py did) leaves stray brackets/quotes glued onto
    the first/last items and breaks every intersection/lookup.
    """

    text = clean(value)
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            return [str(item).strip() for item in parsed]
        except (ValueError, SyntaxError):
            text = text.strip("[]")

    return [item.strip().strip("'\"") for item in text.split(",") if item.strip()]


def split_items(value) -> set:
    """Lowercased token set from a skill field (list-literal or plain comma text)."""
    return {item.lower() for item in parse_list_field(value) if item}


def get_tokens(row, cols) -> set:
    tokens = set()
    for col in cols:
        tokens |= split_items(row.get(col))
    return tokens


def parse_ratings(row, skill_col: str, rating_col: str) -> dict:
    """
    Pairs a skill list column with its matching ratings column by
    position, e.g. tools=['Jira','Slack'] + tools_ratings=[2,4]
    -> {'jira': 2.0, 'slack': 4.0}.
    """

    skills = [s.lower() for s in parse_list_field(row.get(skill_col))]
    ratings = parse_list_field(row.get(rating_col))

    out = {}
    for i, skill in enumerate(skills):
        if i < len(ratings):
            try:
                out[skill] = float(ratings[i])
            except (ValueError, TypeError):
                out[skill] = 1.0
        else:
            out[skill] = 1.0

    return out


def format_with_ratings(row, skill_col: str, rating_col: str) -> str:
    """
    Merges a skill list with its ratings list into readable text, e.g.
    "Jira (Rating: 2), Slack (Rating: 4)" instead of dumping the raw
    "['Jira', 'Slack']" / "[2, 4]" list-literal strings into the text
    fed to the model.
    """

    skills = parse_list_field(row.get(skill_col))
    ratings = parse_list_field(row.get(rating_col))

    if not skills:
        return ""

    parts = []
    for i, skill in enumerate(skills):
        if i < len(ratings):
            parts.append(f"{skill} (Rating: {ratings[i]})")
        else:
            parts.append(skill)

    return ", ".join(parts)


def format_plain_list(value) -> str:
    """Clean comma-joined list, no ratings (e.g. Other Skills, project Tools)."""
    return ", ".join(parse_list_field(value))


def join_parts(parts):
    return " | ".join([p for p in parts if p])


def build_task_text(row):
    return join_parts([
        f"Project ID: {clean(row.get('project_id'))}",
        f"Project Title: {clean(row.get('project_title'))}",
        f"Domain: {clean(row.get('domain'))}",
        f"Sub Domains: {clean(row.get('sub_domains'))}",
        f"Description: {clean(row.get('description'))}",
        f"Tools: {format_plain_list(row.get('tools'))}",
        f"Technologies: {format_plain_list(row.get('technologies'))}",
        f"Languages: {format_plain_list(row.get('languages'))}",
        f"Tasks: {clean(row.get('tasks'))}",
    ])


def build_employee_text(row):
    return join_parts([
        f"Employee ID: {clean(row.get('employee_id'))}",
        f"Name: {clean(row.get('name'))}",
        f"Age: {clean(row.get('age'))}",
        f"Experience Years: {clean(row.get('experience_years'))}",
        f"Job Domain: {clean(row.get('job_domain'))}",
        f"Job Sub Domain: {clean(row.get('job_sub_domain'))}",
        f"Job Role: {clean(row.get('job_role'))}",
        f"Tools: {format_with_ratings(row, 'tools', 'tools_ratings')}",
        f"Technologies: {format_with_ratings(row, 'technologies', 'technologies_ratings')}",
        f"Languages: {format_with_ratings(row, 'languages', 'languages_ratings')}",
        f"Other Skills: {format_plain_list(row.get('other_skills'))}",
        f"Previous Job Domain: {clean(row.get('previous_job_domain'))}",
        f"Previous Job Sub Domain: {clean(row.get('previous_job_sub_domain'))}",
        f"Previous Job Role: {clean(row.get('previous_job_role'))}",
    ])


def score_employee(task_row, emp_row, emp_tokens) -> float:
    task_tokens = get_tokens(task_row, TASK_SKILL_COLS)
    overlap = len(task_tokens & emp_tokens)

    domain_match = 1 if clean(task_row.get("domain")).lower() in clean(emp_row.get("job_domain")).lower() else 0
    subdomain_match = 1 if clean(task_row.get("sub_domains")).lower() in clean(emp_row.get("job_sub_domain")).lower() else 0

    try:
        exp = float(emp_row.get("experience_years", 0) or 0)
    except (ValueError, TypeError):
        exp = 0.0

    tool_ratings = parse_ratings(emp_row, "tools", "tools_ratings")
    tech_ratings = parse_ratings(emp_row, "technologies", "technologies_ratings")
    lang_ratings = parse_ratings(emp_row, "languages", "languages_ratings")

    rating_bonus = 0.0
    for tok in task_tokens:
        rating_bonus += tool_ratings.get(tok, 0)
        rating_bonus += tech_ratings.get(tok, 0)
        rating_bonus += lang_ratings.get(tok, 0)

    return 4.0 * overlap + 6.0 * domain_match + 4.0 * subdomain_match + 0.25 * exp + 0.5 * rating_bonus


# -----------------------------
# Build triplets
# -----------------------------

def main():
    print("Loading employee dataset...")
    employees = pd.read_csv(EMPLOYEES_FILE)

    print("Loading project dataset...")
    tasks = pd.read_csv(PROJECTS_FILE)

    employees["employee_text"] = employees.apply(build_employee_text, axis=1)
    tasks["task_text"] = tasks.apply(build_task_text, axis=1)

    employee_token_sets = [get_tokens(r, EMPLOYEE_SKILL_COLS) for _, r in employees.iterrows()]

    triplets = []

    for _, task_row in tasks.iterrows():
        scored = []
        for e_idx, emp_row in employees.iterrows():
            sc = score_employee(task_row, emp_row, employee_token_sets[e_idx])
            scored.append((e_idx, sc))

        scored.sort(key=lambda x: x[1], reverse=True)

        positive_ids = [idx for idx, _ in scored[:TOP_POSITIVES]]
        negative_ids = [idx for idx, _ in scored[TOP_POSITIVES:]]

        if not negative_ids:
            continue

        anchor = task_row["task_text"]
        score_lookup = dict(scored)

        for p_idx in positive_ids:
            pos_text = employees.loc[p_idx, "employee_text"]
            neg_pool = negative_ids.copy()
            random.shuffle(neg_pool)
            chosen_negs = neg_pool[: min(NEGATIVES_PER_POSITIVE, len(neg_pool))]

            for n_idx in chosen_negs:
                triplets.append({
                    "project_id": clean(task_row.get("project_id")),
                    "anchor": anchor,
                    "positive": pos_text,
                    "negative": employees.loc[n_idx, "employee_text"],
                    "positive_employee_id": clean(employees.loc[p_idx, "employee_id"]),
                    "negative_employee_id": clean(employees.loc[n_idx, "employee_id"]),
                    "positive_score": score_lookup[p_idx],
                    "negative_score": score_lookup[n_idx],
                })

    triplet_df = pd.DataFrame(triplets)

    # -----------------------------
    # 80/20 split
    # -----------------------------

    train_df, test_df = train_test_split(
        triplet_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    train_df.to_csv(TRAIN_PAIRS_FILE, index=False)
    test_df.to_csv(TEST_PAIRS_FILE, index=False)

    print(f"Employees: {len(employees)}")
    print(f"Projects: {len(tasks)}")
    print(f"Total triplets: {len(triplet_df)}")
    print(f"Train triplets ({1 - TEST_SIZE:.0%}): {len(train_df)} -> {TRAIN_PAIRS_FILE}")
    print(f"Test triplets ({TEST_SIZE:.0%}): {len(test_df)} -> {TEST_PAIRS_FILE}")


if __name__ == "__main__":
    main()