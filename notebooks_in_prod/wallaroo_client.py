import wallaroo

def get_workspace(name, create_if_none=False):
    wl = wallaroo.Client()
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == name:
            workspace = ws
            break
    
    if(workspace is None) and create_if_none:
        workspace = wl.create_workspace(name)
    
    if workspace is None :
        raise Exception(f"Could not find workspace {name}")
        
    return workspace


