# Example orchestrator

from google.cloud import bigquery
import wallaroo
import time

wl = wallaroo.Client()
print(f"in task: {wl.in_task()}")
print(f"task args: {wl.task_args()}")
print(f"hello bigquery! {bigquery.__version__}")
print(f"hello wallaroo! {wallaroo.__version__}")

# Enable shelling into task for REPL debugging
if wl.in_task() and "sleep" in wl.task_args():
    time.sleep(9999999)

ws = wl.get_current_workspace()
print(f"cws: {ws}")
print(f"email: {wl.auth.user_email()}")
print(f"orches: {wl.list_orchestrations()._repr_html_()}")
