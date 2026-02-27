import os, requests

API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID   = os.getenv("GOOGLE_CSE_ID")

params = {
    "key": API_KEY,
    "cx":  CSE_ID,
    "q":   "IPL 2025: Jitesh Sharma breaks MS Dhoni’s record in RCB’s thumping win"
}

r = requests.get("https://www.googleapis.com/customsearch/v1", params=params).json()
print("totalResults:", r.get("searchInformation", {}).get("totalResults"))
for item in r.get("items", []):
    print("-", item["title"])
