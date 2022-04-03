# Testing utils for end-to-end tests

import json
import requests


class Keycloak:
    def __init__(self, host, port, admin_username, admin_password):
        self.host = host
        self.port = port
        self.admin_username = admin_username
        self.admin_password = admin_password

    def get_token(self):
        """Using a hardcoded admin password, obtain a session token from keycloak"""
        url = f"http://{self.host}:{self.port}/auth/realms/master/protocol/openid-connect/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        data = {
            "username": self.admin_username,
            "password": self.admin_password,
            "grant_type": "password",
            "client_id": "admin-cli",
        }
        resp = requests.post(url, headers=headers, data=data)
        assert resp.status_code == 200
        token = resp.json()["access_token"]
        assert len(token) > 800
        self.token = token

    def list_users(self):
        url = f"http://{self.host}:{self.port}/auth/admin/realms/master/users"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"bearer {self.token}",
        }
        data={}
        resp = requests.get(url, headers=headers, data=data)
        return resp

    def create_user(self, username, password, email):
        """Create a keycloak test user. Returns ID."""
        url = f"http://{self.host}:{self.port}/auth/admin/realms/master/users"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"bearer {self.token}",
        }
        payload = {
            "username": username,
            "enabled": "true",
            "emailVerified": "true",
            "email": email,
            "credentials": [
                {
                    "type": "password",
                    "value": password,
                    "temporary": "false",
                }
            ],
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload))
        assert resp.status_code == 201
        return resp.headers["Location"].split("/")[-1]

    def delete_user(self, userid):
        """Remove a keycloak user"""
        url = f"http://{self.host}:{self.port}/auth/admin/realms/master/users/{userid}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"bearer {self.token}",
        }
        resp = requests.delete(url, headers=headers)

class JupyterHub:
    def __init__(self, host: str, port: int, admin_username: str, admin_password: str):
        self.host = host
        self.port = port
        self.admin_username = admin_username
        self.admin_password = admin_password
        self.token = None
    
    def get_token(self):
        res = requests.post(f"http://{self.host}:{self.port}/hub/api/authorizations/token", data=json.dumps({"username": self.admin_username, "password": self.admin_password}))
        res.raise_for_status()
        response = res.json()
        self.token = response["token"]

        return self.token
    
    def start_server(self):
        assert self.token

        res = requests.post(f"http://{self.host}:{self.port}/hub/api/users/{self.admin_username}/server", data="{}", headers=self._auth_header())
        if res.status_code == 400:
            # assume 400 means the server is already started
            return
        res.raise_for_status()

    def _auth_header(self):
        return {"Authorization": "Token {}".format(self.token)}