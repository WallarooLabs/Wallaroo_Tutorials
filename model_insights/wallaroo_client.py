import wallaroo
# import e2elib
# import os
# import json

'''
def login(username, password):
    os.environ["WALLAROO_SDK_CREDENTIALS"] = 'creds.json'
    wl = wallaroo.Client(auth_type="user_password")
    #new_workspace = wl.create_workspace("New Project Workspace")
    #_ = wl.set_current_workspace(new_workspace)
    #wl.get_current_workspace()
    return wl


def create_user(name, password, email):
    x=e2elib.Keycloak("keycloak", "8080", "admin","admin")
    x.get_token()
    x.create_user(name, password, email)
    
    
def create_json():
    import json
    json_string =  {"username": "nina", "password": "helloworld"}
    json_string
    with open('creds.json', 'w') as outfile:
        json.dump(json_string, outfile)
        
        
def demo_preload():
    create_json()
    create_user("nina","helloworld","nina@ex.com")
    create_user("bill","password1","bill@ex.com")
    create_user("susan","password2","susan@ex.com")
    create_user("joe","password3","joe@ex.com")
'''   
    
def get_workspace(name):
    wl = wallaroo.Client()
    for ws in wl.list_workspaces():
        if ws.name() == name:
            return ws

    