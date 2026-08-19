from database.supabase_client import supabase

def get_task(task_id):

    task = (
        supabase
        .table("tasks")
        .select("*")
        .eq("id", task_id)
        .single()
        .execute()
    ).data

    return task
