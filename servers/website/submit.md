---
layout: page
title: Submit
permalink: /submit/
---

## Verification API

Use the API to check whether a Python program encodes a given bitstring. The API is rate-limited and each request takes several seconds to respond.

**Endpoint:** `POST {{ site.api_url }}/verify`

### Request

```json
{
  "cover": "<Python source code as a string>",
  "secret": "<bitstring of 0s and 1s>"
}
```

### Response

```json
{
  "match": true
}
```

`match` is `true` if the decoded bitstring of the program **starts with** the provided `secret`.

---

### Python examples

**Check a single program:**

```python
import requests

API_URL = "{{ site.api_url }}"

code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""

resp = requests.post(
    f"{API_URL}/verify",
    json={"cover": code, "secret": "0010"},
)
print(resp.json())  # {"match": true} or {"match": false}
```

**Binary-search a single bit position:**

```python
import requests

API_URL = "{{ site.api_url }}"

def check_prefix(code: str, prefix: str) -> bool:
    resp = requests.post(
        f"{API_URL}/verify",
        json={"cover": code, "secret": prefix},
    )
    resp.raise_for_status()
    return resp.json()["match"]

code = open("mystery_program.py").read()

# Recover bits one at a time
known = ""
for _ in range(20):
    if check_prefix(code, known + "0"):
        known += "0"
    elif check_prefix(code, known + "1"):
        known += "1"
    else:
        break  # no more bits encoded
    print(f"Recovered so far: {known}")
```

Note: each request takes 4-16 seconds. Plan accordingly.
