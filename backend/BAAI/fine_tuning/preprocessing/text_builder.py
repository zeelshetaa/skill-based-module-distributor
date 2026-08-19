"""
text_builder.py

Converts employee and project records into natural language text
for BAAI/bge-base-en-v1.5 embedding generation.
"""

import ast
from typing import Dict


def _parse_list(value) -> list:
    """
    Safely turn a Python-list-literal string, e.g. "['Python', 'CUDA']",
    into a real list. Falls back to a plain comma-split for values that
    aren't actually list literals, and to [] for empty/NaN values.

    This must be used instead of a naive str.split(",") -- splitting the
    raw string directly leaves stray brackets/quotes glued onto the
    first and last items (e.g. "['python'" and "'cuda']").
    """

    if value is None:
        return []

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            return [str(item).strip() for item in parsed]
        except (ValueError, SyntaxError):
            text = text.strip("[]")

    return [item.strip().strip("'\"") for item in text.split(",") if item.strip()]


def _format_with_ratings(skills_value, ratings_value) -> str:
    """
    Merge a skills list with its matching ratings list into a clean,
    human-readable string: "django (Rating: 2), express.js (Rating: 4)".

    Both lists are parsed with _parse_list first so brackets/quotes never
    leak into the output. If the lists are different lengths (bad data),
    skills without a matching rating are shown without one rather than
    crashing or misaligning.
    """

    skills = _parse_list(skills_value)
    ratings = _parse_list(ratings_value)

    if not skills:
        return ""

    parts = []
    for i, skill in enumerate(skills):
        if i < len(ratings):
            parts.append(f"{skill} (Rating: {ratings[i]})")
        else:
            parts.append(skill)

    return ", ".join(parts)


def _format_plain_list(value) -> str:
    """Clean comma-joined list, no ratings involved (e.g. Other Skills)."""
    return ", ".join(_parse_list(value))


class TextBuilder:

    @staticmethod
    def build_employee_text(employee: Dict) -> str:
        """
        Convert one employee record into descriptive text.
        """

        languages = _format_with_ratings(
            employee.get("languages", ""), employee.get("languages_ratings", "")
        )
        technologies = _format_with_ratings(
            employee.get("technologies", ""), employee.get("technologies_ratings", "")
        )
        tools = _format_with_ratings(
            employee.get("tools", ""), employee.get("tools_ratings", "")
        )
        other_skills = _format_plain_list(employee.get("other_skills", ""))

        text = f"""
Employee Name: {employee.get('name', '')}.

Current Role: {employee.get('job_role', '')}.

Experience: {employee.get('experience_years', 0)} years.

Current Domain:
{employee.get('job_domain', '')}

Current Sub Domain:
{employee.get('job_sub_domain', '')}

Programming Languages:
{languages}

Technologies:
{technologies}

Tools:
{tools}

Other Skills:
{other_skills}

Previous Domain:
{employee.get('previous_job_domain', '')}

Previous Sub Domain:
{employee.get('previous_job_sub_domain', '')}

Previous Role:
{employee.get('previous_job_role', '')}
"""

        return " ".join(text.split())


    @staticmethod
    def build_project_text(project: Dict) -> str:
        """
        Convert one project into descriptive text.
        """

        languages = _format_plain_list(project.get("languages", ""))
        technologies = _format_plain_list(project.get("technologies", ""))
        tools = _format_plain_list(project.get("tools", ""))

        text = f"""
Project Title:
{project.get('project_title', '')}

Project Domain:
{project.get('domain', '')}

Project Sub Domain:
{project.get('sub_domains', '')}

Project Description:
{project.get('description', '')}

Required Programming Languages:
{languages}

Required Technologies:
{technologies}

Required Tools:
{tools}

Tasks:
{project.get('tasks', '')}
"""

        return " ".join(text.split())
    
if __name__ == "__main__":
    import pandas as pd

    employee_df = pd.read_csv("../../datasets/employee_dataset.csv")

    employee = employee_df.iloc[0].to_dict()

    print(TextBuilder.build_employee_text(employee))