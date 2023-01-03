## Enablement Management

Enablement Management allows users to see what Wallaroo features have been activated.

* [List Enablement Features](#list-enablement-features)

### List Enablement Features

Lists the enablement features for the Wallaroo instance.

* **PARAMETERS**
  * null:  An empty set `{}`
* **RETURNS**
  * **features** - (*string*): Enabled features.
  * **name** - (*string*): Name of the Wallaroo instance.
  * **is_auth_enabled** - (*bool*): Whether authentication is enabled.

```python
# List enablement features

apiRequest = "/features/list"

data = {
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    {'features': {'plateau': 'true'},
     'name': 'Wallaroo Dev',
     'is_auth_enabled': True}

