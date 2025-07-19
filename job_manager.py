import redis
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from config import settings
from models import JobStatus, JobStatusResponse

class JobManager:
    def __init__(self):
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.job_prefix = "framepack:job:"
        self.queue_key = "framepack:queue"
        self.active_jobs_key = "framepack:active"
        
    def create_job(self, job_data: dict) -> str:
        """Create a new job and add it to the queue"""
        job_id = str(uuid.uuid4())
        
        job_info = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "progress": 0.0,
            "message": "Job created and queued",
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error_message": None,
            "estimated_time_remaining": None,
            "current_step": None,
            "total_steps": job_data.get("steps", 25),
            "generated_frames": 0,
            "video_length": 0.0,
            "request_data": job_data
        }
        
        # Store job info
        self.redis_client.setex(
            f"{self.job_prefix}{job_id}",
            settings.JOB_TIMEOUT,
            json.dumps(job_info)
        )
        
        # Add to queue
        self.redis_client.lpush(self.queue_key, job_id)
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job information"""
        job_data = self.redis_client.get(f"{self.job_prefix}{job_id}")
        if job_data:
            return json.loads(job_data)
        return None
    
    def update_job(self, job_id: str, updates: dict):
        """Update job information"""
        job_data = self.get_job(job_id)
        if job_data:
            job_data.update(updates)
            self.redis_client.setex(
                f"{self.job_prefix}{job_id}",
                settings.JOB_TIMEOUT,
                json.dumps(job_data)
            )
    
    def start_job(self, job_id: str):
        """Mark job as started"""
        self.update_job(job_id, {
            "status": JobStatus.PROCESSING,
            "started_at": datetime.utcnow().isoformat(),
            "message": "Job processing started"
        })
        
        # Add to active jobs
        self.redis_client.sadd(self.active_jobs_key, job_id)
    
    def complete_job(self, job_id: str, result_data: dict):
        """Mark job as completed"""
        self.update_job(job_id, {
            "status": JobStatus.COMPLETED,
            "completed_at": datetime.utcnow().isoformat(),
            "progress": 100.0,
            "message": "Job completed successfully",
            "result_data": result_data
        })
        
        # Remove from active jobs
        self.redis_client.srem(self.active_jobs_key, job_id)
    
    def fail_job(self, job_id: str, error_message: str):
        """Mark job as failed"""
        self.update_job(job_id, {
            "status": JobStatus.FAILED,
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": error_message,
            "message": f"Job failed: {error_message}"
        })
        
        # Remove from active jobs
        self.redis_client.srem(self.active_jobs_key, job_id)
    
    def cancel_job(self, job_id: str):
        """Cancel a job"""
        job_data = self.get_job(job_id)
        if job_data:
            if job_data["status"] in [JobStatus.PENDING, JobStatus.PROCESSING]:
                self.update_job(job_id, {
                    "status": JobStatus.CANCELLED,
                    "completed_at": datetime.utcnow().isoformat(),
                    "message": "Job cancelled by user"
                })
                
                # Remove from queue and active jobs
                self.redis_client.lrem(self.queue_key, 0, job_id)
                self.redis_client.srem(self.active_jobs_key, job_id)
                return True
        return False
    
    def update_progress(self, job_id: str, progress: float, message: str = None, 
                       current_step: int = None, generated_frames: int = None,
                       video_length: float = None, estimated_time_remaining: int = None):
        """Update job progress"""
        updates = {"progress": progress}
        
        if message:
            updates["message"] = message
        if current_step is not None:
            updates["current_step"] = current_step
        if generated_frames is not None:
            updates["generated_frames"] = generated_frames
        if video_length is not None:
            updates["video_length"] = video_length
        if estimated_time_remaining is not None:
            updates["estimated_time_remaining"] = estimated_time_remaining
            
        self.update_job(job_id, updates)
    
    def get_next_job(self) -> Optional[str]:
        """Get the next job from the queue"""
        # Check if we're at max concurrent jobs
        active_count = self.redis_client.scard(self.active_jobs_key)
        if active_count >= settings.MAX_CONCURRENT_JOBS:
            return None
            
        # Get next job from queue
        job_id = self.redis_client.rpop(self.queue_key)
        if job_id:
            return job_id.decode('utf-8') if isinstance(job_id, bytes) else job_id
        return None
    
    def get_queue_size(self) -> int:
        """Get the number of jobs in queue"""
        return self.redis_client.llen(self.queue_key)
    
    def get_active_jobs_count(self) -> int:
        """Get the number of active jobs"""
        return self.redis_client.scard(self.active_jobs_key)
    
    def get_all_jobs(self, page: int = 1, per_page: int = 50) -> List[Dict]:
        """Get all jobs with pagination"""
        # Get all job keys
        pattern = f"{self.job_prefix}*"
        keys = self.redis_client.keys(pattern)
        
        # Calculate pagination
        total = len(keys)
        start = (page - 1) * per_page
        end = start + per_page
        
        jobs = []
        for key in keys[start:end]:
            job_data = self.redis_client.get(key)
            if job_data:
                jobs.append(json.loads(job_data))
        
        # Sort by created_at descending
        jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jobs, total
    
    def cleanup_old_jobs(self):
        """Clean up old completed/failed jobs"""
        cutoff_time = datetime.utcnow() - timedelta(hours=settings.FILE_RETENTION_HOURS)
        cutoff_str = cutoff_time.isoformat()
        
        pattern = f"{self.job_prefix}*"
        keys = self.redis_client.keys(pattern)
        
        for key in keys:
            job_data = self.redis_client.get(key)
            if job_data:
                job_info = json.loads(job_data)
                if (job_info.get('status') in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and
                    job_info.get('completed_at', '') < cutoff_str):
                    self.redis_client.delete(key)
    
    def get_job_status_response(self, job_id: str) -> Optional[JobStatusResponse]:
        """Get job status as response model"""
        job_data = self.get_job(job_id)
        if not job_data:
            return None
            
        return JobStatusResponse(
            job_id=job_data["job_id"],
            status=job_data["status"],
            progress=job_data["progress"],
            message=job_data["message"],
            created_at=job_data["created_at"],
            started_at=job_data.get("started_at"),
            completed_at=job_data.get("completed_at"),
            error_message=job_data.get("error_message"),
            estimated_time_remaining=job_data.get("estimated_time_remaining"),
            current_step=job_data.get("current_step"),
            total_steps=job_data.get("total_steps"),
            generated_frames=job_data.get("generated_frames"),
            video_length=job_data.get("video_length")
        )

# Global job manager instance
job_manager = JobManager()