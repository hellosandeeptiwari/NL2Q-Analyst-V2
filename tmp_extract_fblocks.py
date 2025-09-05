p = r"c:\Users\SandeepTiwari\NL2Q Agent\backend\orchestrators\dynamic_agent_orchestrator.py"
s = open(p, encoding='utf-8', errors='replace').read()
blocks = []
start = 0
while True:
    i = s.find('f"""', start)
    if i == -1:
        break
    j = s.find('"""', i+4)
    if j == -1:
        # no closing triple quote
        j = min(len(s), i+400)
    block = s[i:j+3]
    ln = s.count('\n', 0, i)+1
    blocks.append((ln, block))
    start = j+3

for ln, block in blocks:
    print('--- f""" block starting at line', ln, '---')
    print(block[:800])
    print('\n')
