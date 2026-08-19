from datetime import datetime
import math
import os
from supabase import create_client, Client
# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL or Key not found in environment variables.")
supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)
WORKING_HOURS_PER_WEEK = 48
WORKING_HOURS_PER_DAY = 8
WORKING_DAYS_PER_WEEK = 6
MAXIMUM_ACTIVE_TASKS = 10
# -----------------------------
# TOTAL WEEKLY HOURS
# -----------------------------
def calculate_total_weekly_hours(starting_date, deadline_date):
    total_days = (deadline_date - starting_date).days + 1
    weeks = math.ceil(
        total_days / WORKING_DAYS_PER_WEEK
    )
    return weeks * WORKING_HOURS_PER_WEEK

#parita 18-07
# -----------------------------
# ACTIVE TASK COUNT
# -----------------------------
def estimate_active_tasks_from_supabase(emp_id):
    try:
        response = (
            supabase
            .table("task_assignment")
            .select("id", count="exact")
            .eq("emp_id", emp_id)
            .in_(
                "status",
                [
                    "Assigned",
                    "In Process",
                    "Extended deadline"
                ]
            )
            .execute()
        )

        return response.count or 0

    except Exception as e:
        print(f"Error fetching active tasks: {e}")
        return 0
    
#parita 18-07
# -----------------------------
# FETCH TASK COMPLEXITY & PRIORITY
# -----------------------------
def get_task_details(task_id):
    try:
        response = (
            supabase
            .table("tasks")
            .select(
                "complexity, priority"
            )
            .eq("id", task_id)
            .single()
            .execute()
        )
        return response.data or {}
    except Exception as e:
        print(
            f"Task details error: {e}"
        )
        return {}
# -----------------------------
# COMPLEXITY SCORE
# -----------------------------
def calculate_complexity_score(complexity):
    if not complexity:
        return 50
    complexity = complexity.lower()
    score_mapping = {
        "low": 30,
        "medium": 60,
        "high": 85,
        "very high": 100
    }
    return score_mapping.get(
        complexity,
        50
    )
# -----------------------------
# PRIORITY SCORE
# -----------------------------
def calculate_priority_score(priority):
    if not priority:
        return 50
    priority = priority.lower()
    score_mapping = {
        "low": 30,
        "medium": 60,
        "high": 80,
        "critical": 100
    }
    return score_mapping.get(
        priority,
        50
    )

    
#parita 18-07
# -----------------------------
# AVAILABILITY
# -----------------------------
def calculate_availability(
        total_weekly_hours,
        total_task_hours
):
    free_hours = max(
        total_weekly_hours - total_task_hours,
        0
    )

    availability_score = (
        free_hours / total_weekly_hours
    ) * 100 if total_weekly_hours > 0 else 0

    return {
        "availability_score": round(
            availability_score,
            2
        ),

        "free_hours": round(
            free_hours,
            2
        ),

        "weekly_hours": total_weekly_hours
    }
# -----------------------------
# WORKLOAD SCORE
# -----------------------------
def calculate_workload(
        total_assigned_days,
        total_weekly_hours,
        active_tasks,
        complexity_score,
        priority_score
):

    total_task_hours = (
        total_assigned_days *
        WORKING_HOURS_PER_DAY
    )

    task_load_score = min(
        (
            active_tasks /
            MAXIMUM_ACTIVE_TASKS
        ) * 100,
        100
    )

    hours_utilization = min(
        (
            total_task_hours /
            total_weekly_hours
        ) * 100,
        100
    )

    workload_score = (
        hours_utilization * 0.50 +
        task_load_score * 0.20 +
        complexity_score * 0.20 +
        priority_score * 0.10
    )

    return {

        "total_task_hours": total_task_hours,

        "workload_score": round(
            workload_score,
            2
        ),

        "hours_utilization": round(
            hours_utilization,
            2
        ),

        "task_load_score": round(
            task_load_score,
            2
        ),

        "complexity_score": complexity_score,

        "priority_score": priority_score
    }

# -----------------------------
# RESOURCE BALANCE
# -----------------------------
def calculate_resource_balance(
        availability_score,
        workload_score
):

    remaining_capacity = max(
        100 - workload_score,
        0
    )

    resource_balancing_score = (
        availability_score * 0.50
        +
        remaining_capacity * 0.50
    )

    return round(
        resource_balancing_score,
        2
    )

#parita 18-07
# -----------------------------
# FINAL PIPELINE
# -----------------------------
def calculate_employee_scores(
        starting_date,
        deadline,
        skill_matching_score,
        emp_id,
        task_id
):
    starting_date = datetime.strptime(
        starting_date,
        "%Y-%m-%d"
    ).date()
    deadline_date = datetime.strptime(
        deadline,
        "%Y-%m-%d"
    ).date()
    if deadline_date <= starting_date:
        raise ValueError(
            "Deadline must be greater than Starting Date."
        )
    total_days = (
        deadline_date -
        starting_date
    ).days + 1
    total_weekly_hours = calculate_total_weekly_hours(
        starting_date,
        deadline_date
    )
    # -----------------------------
    # TASK COMPLEXITY & PRIORITY
    # -----------------------------
    task_details = get_task_details(task_id)

    # Actual values from Supabase
    complexity = task_details.get("complexity", "")
    priority = task_details.get("priority", "")

    # Convert to numerical scores   
    complexity_score = calculate_complexity_score(complexity)
    priority_score = calculate_priority_score(priority)
    # -----------------------------
    # ACTIVE TASKS
    # -----------------------------
    # Current active tasks from task_assignment
    current_active_tasks = estimate_active_tasks_from_supabase(
        emp_id
    )

    # After assigning the current task
    active_tasks = current_active_tasks + 1

    # current active task from employee
    current_assigned_days = get_total_assigned_days(emp_id)

    total_assigned_days = (
        current_assigned_days +
        total_days
    )
    # -----------------------------
    # WORKLOAD
    # -----------------------------
    workload = calculate_workload(
        total_assigned_days,
        total_weekly_hours,
        active_tasks,
        complexity_score,
        priority_score
    )

     # -----------------------------
    # AVAILABILITY
    # -----------------------------
    availability = calculate_availability(
        total_weekly_hours,
        workload["total_task_hours"] # parita 18-07
    )

    # -----------------------------
    # RESOURCE BALANCING
    # -----------------------------
    resource_score = calculate_resource_balance(
        availability["availability_score"],
        workload["workload_score"]
    )
    # -----------------------------
    # SKILL SCORE NORMALIZATION
    # -----------------------------
    if skill_matching_score <= 1:
        skill_matching_score *= 100
    # -----------------------------
    # FINAL SCORE
    # -----------------------------
    final_workload_score = (
        skill_matching_score * 0.70
        +
        resource_score * 0.30
    )
    return {

        "deadline": deadline,
        "total_days": total_days,

        "total_task_hours":
            workload["total_task_hours"],

        "total_weekly_available_hours":
            total_weekly_hours,

        "free_hour_before_deadline":
            availability["free_hours"],

        "availability_score":
            availability["availability_score"],

        "active_tasks":
            active_tasks,

        "task_load":
            workload["task_load_score"],

        "hours_utilization":
            workload["hours_utilization"],

        # Actual values (save these to Supabase)
        "complexity":
            complexity,

        "priority":
            priority,

        # Numerical scores (used only for calculations)
        "complexity_score":
            complexity_score,

        "priority_score":
            priority_score,

        "workload_score":
            workload["workload_score"],

        "resource_balancing_score":
            resource_score,

        "skill_matching_score":
            round(skill_matching_score, 2),

        "final_workload_score":
            round(final_workload_score, 2)
    }

#parita 18-07
# -----------------------------
# FETCH TOTAL ASSIGNED DAYS
# -----------------------------
def get_total_assigned_days(emp_id):
    try:
        response = (
            supabase
            .table("task_assignment")
            .select("total_days")
            .eq("emp_id", emp_id)
            .eq("status", "Assigned")
            .execute()
        )

        assigned_days = sum(
            task.get("total_days") or 0
            for task in response.data
        )

        return assigned_days

    except Exception as e:
        print(f"Assigned days error: {e}")
        return 0
    #parita 18-07
