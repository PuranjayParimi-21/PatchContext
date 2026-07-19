import json
import os

for filename in ['commits.json', 'prs.json', 'issues.json']:
    path = os.path.join('data', filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"{filename}: {len(data)} items")
    else:
        print(f"{filename} does not exist")
