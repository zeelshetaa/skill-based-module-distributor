from database.supabase_client import supabase


def get_all_employees():

    profiles = (
        supabase
        .table("employee_profiles")
        .select("*")
        .execute()
    ).data

    employees = []

    for profile in profiles:

        emp_id = profile["emp_id"]

        skills = (
            supabase
            .table("employee_skills")
            .select("*")
            .eq("emp_id", emp_id)
            .single()
            .execute()
        ).data

        features = (
            supabase
            .table("employee_features")
            .select("*")
            .eq("emp_id", emp_id)
            .single()
            .execute()
        ).data

        workload = (
            supabase
            .table("employee_workload")
            .select("*")
            .eq("emp_id", emp_id)
            .single()
            .execute()
        ).data

        employees.append({

            "emp_id": emp_id,

            "name": profile["name"],

            "role": profile["role"],

            "experience_years": profile["experience_years"],

            **skills,

            **features,

            **workload

        })

    return employees
