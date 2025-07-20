# Celery Configuration
from config import settings

# Broker settings
broker_url = settings.CELERY_BROKER_URL
result_backend = settings.CELERY_RESULT_BACKEND

# Task settings
task_serializer = 'json'
accept_content = ['json']
result_serializer = 'json'
timezone = 'UTC'
enable_utc = True

# Worker settings
worker_prefetch_multiplier = 1
task_acks_late = True
worker_max_tasks_per_child = 1

# Task routing
task_routes = {
    'framepack_worker.process_job': {'queue': 'video_generation'},
}

# Task time limits
task_soft_time_limit = 3600  # 1 hour
task_time_limit = 3900  # 1 hour 5 minutes

# Result backend settings
result_expires = 3600  # 1 hour