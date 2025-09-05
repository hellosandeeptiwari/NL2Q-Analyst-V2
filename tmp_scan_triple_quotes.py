p = r"c:\Users\SandeepTiwari\NL2Q Agent\backend\orchestrators\dynamic_agent_orchestrator.py"
with open(p, encoding='utf-8', errors='replace') as f:
    s = f.read()
    td = s.count('"""')
    ts = s.count("'''")
    print(f"triple_double={td}, triple_single={ts}\n")
    for i, line in enumerate(s.splitlines(), 1):
        if '"""' in line or "'''" in line:
            print(f"{i}: {line.rstrip()}")
