import json
import os
import re
import time





datas_line = []

filel = open("data.json", "w", encoding="utf-8")
file1 = open("времфайл", "w", encoding="utf-8")

with open("dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        line = json.loads(line)
        q = line['input']
        d = line['instruction']
        a = line['output']
        print(q)
        if not any(c.isdigit() for c in q) and not any(c.isdigit() for c in a) and not any(c.isdigit() for c in d):
            q = re.sub(r'[^a-zA-Z\s]', '', q)
            a = re.sub(r'[^a-zA-Z\s]', '', a)
            d = re.sub(r'[^a-zA-Z\s]', '', d)
            q = q.lower()
            a = a.lower()
            d = d.lower()
            qes = (d if d != '' else '') + (' ' if (d != '' and q != '') else '') + (q if q != '' else '')
            data = {
                "q": qes,
                "a": a
                
            }
            if qes != '' and a != '':
                filel.write(json.dumps(data, ensure_ascii=False) + "\n")
                file1.write(data["q"]+"\n")
                file1.write(data["a"]+"\n")
            else:
                print("ОПАААА")
            time.sleep(10)































































