from fastapi import APIRouter
from pydantic import BaseModel
from datetime import date
from typing import List
from backend.gemini_services.requirement_extractor import (
    extract_requirements,
)
from database.supabase_client import supabase
router = APIRouter()
# =====================================================
# MODELS
# =====================================================
class TaskModel(BaseModel):
    task_name: str
    description: str
    technologies: List[str]
    tools_and_ide: List[str]
    starting_date: date
    deadline: date
    complexity: str
    priority: str
class UpdateTaskStatus(BaseModel):
    task_id: str
    status: str
    deadline: date
# =====================================================
# CREATE TASK
# =====================================================
# =====================================================
# CREATE TASK
# =====================================================
@router.post("/task")
def create_task(task: TaskModel):
    # ----------------------------------------
    # Convert request into dictionary
    # ----------------------------------------
    task_data = task.model_dump()
    # ----------------------------------------
    # Gemini Requirement Extraction
    # ----------------------------------------
    extracted_data = extract_requirements(task_data)
    # ----------------------------------------
    # Calculate Duration
    # ----------------------------------------
    duration_days = (task.deadline - task.starting_date).days
    if duration_days <= 0:
        return {
            "message": "Deadline must be after Starting Date."
        }
    # ----------------------------------------
    # Build Database Record
    # ----------------------------------------
    data = {
        "task_name": task.task_name,
        "description": task.description,
        "technologies": task.technologies,
        "tools_and_ide": task.tools_and_ide,
        "required_skills":
            extracted_data["required_skills"],
        "duration_days":
            duration_days,
        "starting_date":
            str(task.starting_date),
        "deadline":
            str(task.deadline),
        "complexity":
            task.complexity,
        "priority":
            task.priority,
        "status":
            "Pending"
    }
    result = (
        supabase
        .table("tasks")
        .insert(data)
        .execute()
    )
    return {
        "message": "Task Created Successfully",
        "data": result.data
    }
# =====================================================
# GET ALL ASSIGNED TASKS OF EMPLOYEE
# =====================================================
@router.get("/task-assigned/{emp_id}")
def get_task_assignment(emp_id: int):
    assignments = (
        supabase
        .table("task_assignment")
        .select("*")
        .eq("emp_id", emp_id)
        .order("final_score", desc=True)
        .execute()
    ).data
    if not assignments:
        return []
    response = []
    for assignment in assignments:
        task = (
            supabase
            .table("tasks")
            .select("""
                status,
                deadline,
                priority,
                complexity,
                duration_days,
                starting_date
            """)
            .eq("id", assignment["task_id"])
            .execute()
        ).data
        status = "Assigned"
        deadline = ""
        priority = ""
        complexity = ""
        duration = 0
        starting_date = ""
        if task:
            status = task[0]["status"]
            deadline = task[0]["deadline"]
            priority = task[0]["priority"]
            complexity = task[0]["complexity"]
            duration = task[0]["duration_days"]
            starting_date = task[0]["starting_date"]
        response.append({
            "task_id":
                assignment["task_id"],
            "task_name":
                assignment["task_name"],
            "employee_name":
                assignment["employee_name"],
            "similarity_score":
                assignment["similarity_score"],
            "workload_score":
                assignment["workload_score"],
            "final_score":
                assignment["final_score"],
            "status":
                status,
            "deadline":
                deadline,
            "priority":
                priority,
            "complexity":
                complexity,
            "duration_days":
                duration,
            "starting_date":
                starting_date
        })
    return response
# =====================================================
# ADMIN - VIEW ALL ASSIGNED TASKS
# =====================================================
@router.get("/admin-task-assignments")
def get_admin_task_assignments():
    assignments = (
        supabase
        .table("task_assignment")
        .select("*")
        .order("final_score", desc=True)
        .execute()
    ).data
    if not assignments:
        return []
    data = []
    for assignment in assignments:
        task = (
            supabase
            .table("tasks")
            .select("""
                priority,
                complexity,
                deadline,
                duration_days
            """)
            .eq("id", assignment["task_id"])
            .execute()
        ).data
        priority = ""
        complexity = ""
        deadline = ""
        duration = ""
        if task:
            priority = task[0]["priority"]
            complexity = task[0]["complexity"]
            deadline = task[0]["deadline"]
            duration = task[0]["duration_days"]
        data.append({
            "employee":
                assignment["employee_name"],
            "task":
                assignment["task_name"],
            "priority":
                priority,
            "complexity":
                complexity,
            "deadline":
                deadline,
            "duration_days":
                duration,
            "similarity":
                assignment["similarity_score"],
            "workload":
                assignment["workload_score"],
            "score":
                assignment["final_score"]
        })
    return data
# =====================================================
# UPDATE TASK STATUS & DEADLINE
# =====================================================
@router.put("/update-task-status")
def update_task_status(data: UpdateTaskStatus):
    # # ----------------------------------------
    # # Update Status & Deadline
    # # ----------------------------------------
    # (
    #     supabase
    #     .table("tasks")
    #     .update({
    #         "status": data.status,
    #         "deadline": str(data.deadline)
    #     })
    #     .eq("id", data.task_id)
    #     .execute()
    # )
    # ----------------------------------------
    # Update Status & Deadline in tasks table
    # ----------------------------------------
    # ----------------------------------------
    # Update Tasks Table
    # ----------------------------------------
    (
        supabase
        .table("tasks")
        .update({
            "status": data.status,
            "deadline": str(data.deadline)
        })
        .eq("id", data.task_id)
        .execute()
    )
    # ----------------------------------------
    # Update Task Assignment Table
    # ----------------------------------------
    (
        supabase
        .table("task_assignment")
        .update({
            "status": data.status
        })
        .eq("task_id", data.task_id)
        .execute()
    )
    # ----------------------------------------
    # In Process
    # ----------------------------------------
    if data.status == "In process":
        return {
            "message": "Task status changed to In process"
        }
    # ----------------------------------------
    # Extend Deadline
    # ----------------------------------------
    if data.status == "Extend deadline":
        return {
            "message": "Deadline extended successfully"
        }
    # ----------------------------------------
    # Completed
    # ----------------------------------------
    if data.status != "Completed":
        return {
            "message": "Task Updated Successfully"
        }
    # Remaining Completed logic...
    # ----------------------------------------
    # Fetch Task
    # ----------------------------------------
    task = (
        supabase
        .table("tasks")
        .select("*")
        .eq("id", data.task_id)
        .single()
        .execute()
    ).data
    if not task:
        return {
            "message": "Task not found."
        }
    # ----------------------------------------
    # Fetch Assignment
    # ----------------------------------------
    assignment = (
        supabase
        .table("task_assignment")
        .select("*")
        .eq("task_id", data.task_id)
        .single()
        .execute()
    ).data
    employee_name = ""
    emp_id = None
    if assignment:
        employee_name = assignment["employee_name"]
        emp_id = assignment["emp_id"]
    # ----------------------------------------
    # Save into task_completed
    # ----------------------------------------
    (
        supabase
        .table("task_completed")
        .insert({
            "task_id": task["id"],
            "task_name": task["task_name"],
            "description": task["description"],
            "technologies": task["technologies"],
            "tools_and_ide": task["tools_and_ide"],
            "required_skills": task["required_skills"],
            "duration_days": task["duration_days"],
            "starting_date": task["starting_date"],
            "deadline": task["deadline"],
            "priority": task["priority"],
            "complexity": task["complexity"],
            "status": "Completed",
            "employee_name": employee_name,
            # "emp_id": emp_id
            "employee_name": employee_name,
        })
        .execute()
    )
    # ----------------------------------------
    # Remove Assignment
    # ----------------------------------------
    (
        supabase
        .table("task_assignment")
        .delete()
        .eq("task_id", data.task_id)
        .execute()
    )
    # ----------------------------------------
    # Remove Employee Workload
    # ----------------------------------------
# (
#     supabase
#     .table("employee_workload")
#     .delete()
#     .eq("task_id", data.task_id)
#     .execute()
# )   
    # ----------------------------------------
    # Update Employee Workload
    # ----------------------------------------
# Delete the completed task workload
    # ----------------------------------------
# Update Employee Workload
# ----------------------------------------

    workload = (
        supabase
        .table("employee_workload")
        .select("active_tasks")
        .eq("emp_id", emp_id)
        .single()
        .execute()
    ).data

    if workload:

        (
            supabase
            .table("employee_workload")
            .update({
                "active_tasks": max(
                    workload["active_tasks"] - 1,
                    0
                )
            })
            .eq("emp_id", emp_id)
            .execute()
        )
    # ----------------------------------------
    # Remove Task
    # ----------------------------------------
    (
        supabase
        .table("tasks")
        .delete()
        .eq("id", data.task_id)
        .execute()
    )
    return {
        "message": "Task Completed Successfully"
    }
