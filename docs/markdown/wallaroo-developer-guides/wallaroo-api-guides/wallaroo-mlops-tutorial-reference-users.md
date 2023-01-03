## Users

Users can be created, activated, and deactivated through the Wallaroo MLOps API.

* [Get Users](#get-users)
* [Invite Users](#invite-users)
* [Deactivate User](#deactivate-user)
* [Activate User](#activate-user)

### Get Users

Users can be retrieved either by their Keycloak user id, or return all users if an empty set `{}` is submitted.

* **Parameters**
  * `{}`: Empty set, returns all users.
  * **user_ids** *Array[Keycloak user ids]*: An array of Keycloak user ids, typically in UUID format.

Example:  The first example will submit an empty set `{}` to return all users, then submit the first user's user id and request only that user's details.

```python
# Get all users

apiRequest = "/users/query"
data = {
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    {'users': {'5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f': {'access': {'impersonate': True,
        'manageGroupMembership': True,
        'manage': True,
        'mapRoles': True,
        'view': True},
       'createdTimestamp': 1669221287375,
       'disableableCredentialTypes': [],
       'email': 'john.hansarick@wallaroo.ai',
       'emailVerified': True,
       'enabled': True,
       'id': '5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'john.hansarick@wallaroo.ai'},
      '941937b3-7dc8-4abe-8bb1-bd23c816421e': {'access': {'view': True,
        'manage': True,
        'manageGroupMembership': True,
        'mapRoles': True,
        'impersonate': True},
       'createdTimestamp': 1669221214282,
       'disableableCredentialTypes': [],
       'emailVerified': False,
       'enabled': True,
       'id': '941937b3-7dc8-4abe-8bb1-bd23c816421e',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'admin'},
      'da7c2f4c-822e-49eb-93d7-a4b90af9b4ca': {'access': {'mapRoles': True,
        'impersonate': True,
        'manage': True,
        'manageGroupMembership': True,
        'view': True},
       'createdTimestamp': 1669654086172,
       'disableableCredentialTypes': [],
       'email': 'kilvin.mitchell@wallaroo.ai',
       'emailVerified': True,
       'enabled': True,
       'id': 'da7c2f4c-822e-49eb-93d7-a4b90af9b4ca',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'kilvin.mitchell@wallaroo.ai'}}}

```python
# Get first user Keycloak id
firstUserKeycloak = list(response['users'])[0]

apiRequest = "/users/query"
data = {
  "user_ids": [
    firstUserKeycloak
  ]
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    {'users': {'5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f': {'access': {'view': True,
        'manage': True,
        'manageGroupMembership': True,
        'mapRoles': True,
        'impersonate': True},
       'createdTimestamp': 1669221287375,
       'disableableCredentialTypes': [],
       'email': 'john.hansarick@wallaroo.ai',
       'emailVerified': True,
       'enabled': True,
       'id': '5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'john.hansarick@wallaroo.ai'}}}

### Invite Users

**IMPORTANT NOTE**:  This command is for Wallaroo Community only.  For more details on user management, see [Wallaroo User Management](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-user-management/).

Users can be invited through `/users/invite`.  When using Wallaroo Community, this will send an invitation email to the email address listed.  Note that the user must not already be a member of the Wallaroo instance, and email addresses must be unique.  If the email address is already in use for another user, the request will generate an error.

* **Parameters**
  * **email** *(REQUIRED string): The email address of the new user to invite.
  * **password** *(OPTIONAL string)*: The assigned password of the new user to invite.  If not provided, the Wallaroo instance will provide the new user a temporary password that must be changed upon initial login.

Example:  In this example, a new user will be invited to the Wallaroo instance and assigned a password.

```python
# invite users
apiRequest = "/users/invite"
data = {
    "email": newUser,
    "password":newPassword
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

### Deactivate User

Users can be deactivated so they can not login to their Wallaroo instance.  Deactivated users do not count against the Wallaroo license count.

* **Parameters**
  * **email** (*REQUIRED string*):  The email address of the user to deactivate.

Example:  In this example, the `newUser` will be deactivated.

```python
# Deactivate users

apiRequest = "/users/deactivate"

data = {
    "email": newUser
}
```

```python
response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    {}

### Activate User

A deactivated user can be reactivated to allow them access to their Wallaroo instance.  Activated users count against the Wallaroo license count.

* **Parameters**
  * **email** (*REQUIRED string*):  The email address of the user to activate.

Example:  In this example, the `newUser` will be activated.

```python
# Activate users

apiRequest = "/users/activate"

data = {
    "email": newUser
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    {}

