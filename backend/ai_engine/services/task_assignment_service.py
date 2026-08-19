from datetime import datetime
from database.supabase_client import supabase
from backend.resource_workload.workload_service import (
    generate_employee_workload,
)
# ==========================================================
# Priority Score
# ==========================================================
def get_priority_score(priority):
    priority_map = {
        "Critical": 100,
        "High": 80,
        "Medium": 60,
        "Low": 40,
        "Very Low": 20
    }
    return priority_map.get(priority, 50)
# ==========================================================
# Complexity Score
# ==========================================================
def get_complexity_score(complexity):
    complexity_map = {
        "Very Easy": 50,
        "Easy": 60,
        "Medium": 75,
        "Hard": 90,
        "Expert": 100
    }
    return complexity_map.get(complexity, 70)
# ==========================================================
# Deadline Urgency Score
# ==========================================================
def get_deadline_score(deadline):
    deadline_date = datetime.strptime(
        deadline,
        "%Y-%m-%d"
    )
    days_left = (
        deadline_date -
        datetime.today()
    ).days
    if days_left <= 2:
        return 100
    elif days_left <= 5:
        return 90
    elif days_left <= 10:
        return 80
    elif days_left <= 15:
        return 70
    elif days_left <= 20:
        return 60
    return 50
# ==========================================================
# Intelligent Task Assignment
# ==========================================================
# ==========================================================
# Intelligent Task Assignment
# ==========================================================
def assign_tasks():

    # ----------------------------------------
    # Load similarity results
    # ----------------------------------------
    matches = (
        supabase
        .table("employee_task_match")
        .select("*")
        .execute()
    ).data

    if not matches:
        print("No employee matches found.")
        return []

    # ----------------------------------------
    # Load Pending Tasks
    # ----------------------------------------
    pending_tasks = (
        supabase
        .table("tasks")
        .select("*")
        .eq("status", "Pending")
        .execute()
    ).data

    if not pending_tasks:
        print("No pending tasks.")
        return []

    final_assignments = []

    # Prevent one employee from receiving
    # multiple NEW tasks in THIS backend run.
    assigned_employees = set()

    # ======================================================
    # LOOP TASK BY TASK
    # ======================================================
    for task in pending_tasks:

        # ----------------------------------------
        # All employee matches for THIS task
        # ----------------------------------------
        task_candidates = []

        for row in matches:

            if row["task_id"] != task["id"]:
                continue

    # Ignore employees with similarity below 20%
            if float(row["similarity_score"]) < 20:
                continue

        task_candidates.append(row)

        if not task_candidates:
            continue

        # Highest similarity first
        task_candidates.sort(
            key=lambda x: float(x["similarity_score"]),
            reverse=True
        )

        assigned = False

        # ----------------------------------------
        # Find best available employee
        # ----------------------------------------
        for row in task_candidates:

            # Already received a task during
            # THIS backend execution
            if row["emp_id"] in assigned_employees:
                continue

            try:

                workload = generate_employee_workload(
                    emp_id=row["emp_id"],
                    task_id=task["id"],
                    similarity_score=float(row["similarity_score"])
                )

            except Exception as e:

                print(
                    f"Workload Error ({row['name']}): {e}"
                )

                continue

            priority_score = get_priority_score(
                task["priority"]
            )

            complexity_score = get_complexity_score(
                task["complexity"]
            )

            urgency_score = get_deadline_score(
                task["deadline"]
            )

            final_score = (

                float(row["similarity_score"]) * 0.45

                +

                workload["workload_score"] * 0.25

                +

                priority_score * 0.15

                +

                urgency_score * 0.10

                +

                complexity_score * 0.05

            )

            # ----------------------------------------
            # Update task
            # ----------------------------------------
            (
                supabase
                .table("tasks")
                .update({
                    "status": "Assigned"
                })
                .eq("id", task["id"])
                .execute()
            )

            final_assignments.append({

                "emp_id": row["emp_id"],

                "employee_name": row["name"],

                "task_id": task["id"],

                "task_name": task["task_name"],

                "similarity_score": float(row["similarity_score"]),

                "workload_score": workload["workload_score"],

                "priority": task["priority"],

                "complexity": task["complexity"],

                "deadline": task["deadline"],

                "priority_score": priority_score,

                "complexity_score": complexity_score,

                "urgency_score": urgency_score,

                "final_score": round(final_score, 2)

            })

            assigned_employees.add(row["emp_id"])

            print(
                f"✓ {row['name']} → {task['task_name']} "
                f"Final Score : {round(final_score,2)}"
            )

            assigned = True
            break

        if not assigned:
            print(
                f"No available employee for {task['task_name']}"
            )

    print("\n" + "=" * 70)
    print("SMART TASK ASSIGNMENT COMPLETED")
    print("=" * 70)
    print(f"Assignments : {len(final_assignments)}")
    print("=" * 70)

    return final_assignments
