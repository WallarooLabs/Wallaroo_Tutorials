
import wallaroo
import os

 

wl = wallaroo.Client()
ws = wl.list_workspaces()
for w in ws:
    if w.name() == 'computer-vision':
        wl.set_current_workspace(w)

for d in wl.list_deployments():
    d.undeploy()
