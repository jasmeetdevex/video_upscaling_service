# TODO: Run upscale_video_task in subprocess without Celery

- [x] Modify upscale_video_task function: remove Celery decorator, make it a regular function, add file logging, initialize mongo inside the function
- [x] Update the /api/upscale/task/<task_id> endpoint to use multiprocessing.Process to start the task in background
- [x] Ensure proper error handling and logging in the subprocess
- [x] Test the changes to ensure API returns immediately and task runs in background with logs
