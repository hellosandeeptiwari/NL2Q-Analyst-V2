p = r"c:\Users\SandeepTiwari\NL2Q Agent\backend\orchestrators\dynamic_agent_orchestrator.py"
s = open(p, encoding='utf-8', errors='replace').read()
occ = []
start = 0
while True:
    i = s.find('"""', start)
    if i == -1:
        break
    ln = s.count('\n', 0, i) + 1
    occ.append((i, ln))
    start = i + 3
print('Found', len(occ), 'triple-double occurrences')
# Pair them sequentially
stack = []
for idx, (pos, ln) in enumerate(occ, 1):
    # naive: assume quotes alternate open/close
    if not stack or stack[-1][2] == 'closed':
        stack.append([idx, pos, 'open', ln])
    else:
        # close previous
        stack[-1][2] = 'closed'
        stack.append([idx, pos, 'closed', ln])

# Find first item marked open without closed partner
open_items = [item for item in stack if item[2] == 'open']
if open_items:
    item = open_items[-1]
    print('\nUnmatched occurrence index (1-based):', item[0], 'line:', item[3])
    pos = item[1]
    # print surrounding context
    start = max(0, pos - 200)
    end = min(len(s), pos + 200)
    ctx = s[start:end]
    print('\nContext around unmatched triple-quote:')
    print(ctx)
else:
    print('\nNo unmatched occurrences found by naive pairing (even count).')
    
# Also list last 10 occurrences with lines
print('\nLast 10 occurrences (index, line):')
for idx, (pos, ln) in enumerate(occ[-10:], start=len(occ)-9):
    print(idx, ln)
