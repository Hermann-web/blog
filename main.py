import os
import yaml

def define_env(env):
    """
    Hook for mkdocs-macros-plugin to define variables.
    """
    # Load projects from blog/projects.yml
    config_dir = os.path.dirname(env.conf['config_file_path'])
    projects_file = os.path.join(config_dir, "projects.yml")
    
    projects_data = {}
    if os.path.exists(projects_file):
        with open(projects_file, 'r') as f:
            projects_data = yaml.safe_load(f).get('projects', {})

    # Load modules from blog/python_modules.yml
    modules_file = os.path.join(config_dir, "python_modules.yml")
    modules_data = {}
    if os.path.exists(modules_file):
        with open(modules_file, 'r') as f:
            modules_data = yaml.safe_load(f).get('modules', {})

    resume_data = {
        "personal_info": {
            "name": "Hermann Agossou",
            "title": "Full-stack Web Developer & Data Scientist",
            "location": "Casablanca, Morocco",
            "email": "agossouhermann7@gmail.com"
        },
        # "experiences": {
        #     "MAGNETIC_REFRIGERATION": {
        #         "title": "Scientific Project: Magnetic Refrigeration",
        #         "company": "Mohammed V University in Rabat",
        #         "date": "Oct 2020 — Feb 2021",
        #         "location": "Casablanca, Morocco",
        #         "bullet_points": [
        #             "Conducted a comprehensive review of magnetic refrigeration research (material properties and system design)",
        #             "Analyzed magnetocaloric materials and their applications in modern refrigeration systems",
        #             "Explored historical developments and current challenges in the field of magnetic refrigeration"
        #         ]
        #     },
        #     "MAGHREB_STEEL_INTERN": {
        #         "title": "Internship: Digital Transformation",
        #         "company": "Maghreb Steel",
        #         "date": "Aug 2020 — Sep 2020",
        #         "location": "Remote, Casablanca",
        #         "bullet_points": [
        #             "Conducted cloud service benchmarking for virtualisation purposes (GCP, IBM Cloud, AWS, Microsoft Azure)",
        #             "Designed step-by-step guides for cloud data migration and ML model deployment for Azure ML case"
        #         ]
        #     }
        # },
        "projects": projects_data, # Use data from projects.yml
        "modules": modules_data, # Use data from python_modules.yml
        "education": {
            "ENGINEERING": {
                "degree": "Engineering Degree in Computer Science",
                "school": "Mohammed V University in Rabat",
                "date": "2018 — 2021"
            }
        }
    }

    # Expose to macros (for markdown)
    env.variables['resume'] = resume_data
    
    # Expose to mkdocs config (for templates)
    env.conf['resume'] = resume_data
