import json
import os
import re
import time
from tqdm import tqdm




datas_line = []

filel = open("data.json", "w", encoding="utf-8")
file1 = open("времфайл", "w", encoding="utf-8")

with open("dataset.jsonl", "r", encoding="utf-8") as f:
    for line in tqdm(f):
        line = line.strip()
        line = json.loads(line)
        q = line['data'][0]
        a = line['data'][1]
        #print(q)



        q = re.sub(r'[^a-zA-Z0-9 \n]', '', q)
        a = re.sub(r'[^a-zA-Z0-9 \n]', '', a)
        q = q.lower()
        a = a.lower()
        #qes = (d if d != '' else '') + (' ' if (d != '' and q != '') else '') + (q if q != '' else '')
        data = {
            "q": q,
            "a": a

        }
        if q != '' and a != '':
            if not ("as an ai" in a):
                filel.write(json.dumps(data, ensure_ascii=False) + "\n")
                file1.write(data["q"]+"\n")
                file1.write(data["a"]+"\n")
                #print(data["q"])
                #print("-----------")
                #print(data["a"])
            else:
                print("курва")
        else:
            print("ОПАААА")































































