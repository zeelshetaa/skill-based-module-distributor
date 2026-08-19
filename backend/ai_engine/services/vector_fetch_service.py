from database.supabase_client import supabase
import ast


# --------------------------------------------------------
# Fetch Employee Vectors
# --------------------------------------------------------

def get_employee_vectors():

    response = (
        supabase
        .table("employee_vector")
        .select("*")
        .execute()
    )

    employee_vectors = response.data

    for employee in employee_vectors:

        embedding = employee["embedding"]

        if isinstance(embedding, str):
            employee["embedding"] = ast.literal_eval(embedding)

    return employee_vectors


# --------------------------------------------------------
# Fetch Pending Task Vectors
# --------------------------------------------------------

def get_task_vectors():

    # ----------------------------------------
    # Step 1 : Fetch Pending Task IDs
    # ----------------------------------------

    pending_response = (
        supabase
        .table("tasks")
        .select("id")
        .eq("status", "Pending")
        .execute()
    )

    pending_tasks = pending_response.data

    if not pending_tasks:
        return []

    pending_ids = [
        task["id"]
        for task in pending_tasks
    ]

    # ----------------------------------------
    # Step 2 : Fetch Only Pending Embeddings
    # ----------------------------------------

    vector_response = (
        supabase
        .table("task_vector")
        .select("*")
        .in_("task_id", pending_ids)
        .execute()
    )

    task_vectors = vector_response.data

    for task in task_vectors:

        embedding = task["embedding"]

        if isinstance(embedding, str):
            task["embedding"] = ast.literal_eval(embedding)

    return task_vectors