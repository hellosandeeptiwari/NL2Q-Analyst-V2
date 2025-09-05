import sys
import traceback
from pathlib import Path

p = Path(r"c:\Users\SandeepTiwari\NL2Q Agent\backend\orchestrators\dynamic_agent_orchestrator.py")
if not p.exists():
    print(f"ERROR: File not found: {p}")
    sys.exit(2)

s = p.read_text(encoding='utf-8', errors='replace')

# Count triple-quote occurrences
td = s.count('"""')
ts = s.count("'''")
print(f"Triple-double quotes: {td}, Triple-single quotes: {ts}\n")

# List lines containing triple quotes (first 60)
print("Lines with triple-double or triple-single quotes (first 100 shown):")
for i, line in enumerate(s.splitlines(), 1):
    if '"""' in line or "'''" in line:
        print(f"{i}: {line.rstrip()}")

print('\nAttempting to compile the file...')
try:
    compile(s, str(p), 'exec')
    print('✅ Compilation succeeded: No SyntaxError')
    sys.exit(0)
except SyntaxError as e:
    print('❌ SyntaxError detected during compilation:')
    print(f'  Message: {e.msg}')
    print(f'  Filename: {e.filename}')
    print(f'  Line: {e.lineno}, Offset: {e.offset}')
    print(f'  Text: {e.text.strip() if e.text else None}')
    # Show surrounding context
    lines = s.splitlines()
    start = max(0, (e.lineno or 1) - 6)
    end = min(len(lines), (e.lineno or 1) + 5)
    print('\nContext:')
    for ln in range(start, end):
        marker = '>>' if (ln + 1) == e.lineno else '  '
        print(f"{marker} {ln+1:4}: {lines[ln].rstrip()}")
    # Also show last 200 chars of file to see if file ends abruptly
    print('\nFile tail (last 400 chars):')
    print(s[-400:])
    sys.exit(1)
except Exception:
    print('❌ Unexpected error while compiling:')
    traceback.print_exc()
    sys.exit(3)
