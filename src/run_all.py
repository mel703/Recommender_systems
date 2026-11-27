"""
Общий прогон ЛР 4 (делает Мелине).

Запуск (из папки src, когда все прислали свои скрипты):
    python run_all.py

Что делает:
    1. По очереди запускает все скрипты (.py) и ноутбуки (.ipynb) из src,
       кроме служебных (common.py, run_all.py, example.py).
    2. Весь вывод собирает в reports/obshchiy_progon.txt.
       Числа для отчётов и слайдов берём только оттуда.
    3. Предупреждает, если скрипт сам делит данные (train_test_split или своё условие по split):
       по правилам деление на обучение и проверку берётся только из common.split().
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).parent
REPORT = SRC.parent / 'reports' / 'obshchiy_progon.txt'
SKIP = {'common.py', 'run_all.py', 'example.py'}
FORBIDDEN = ('train_test_split', "['split']")
TMP = SRC / '_notebook_tmp.py'
TIMEOUT_S = 1800


def notebook_to_script(path):
    """Код ячеек ноутбука -> временный .py (строки с % и ! пропускаются, display -> print)."""
    nb = json.loads(path.read_text(encoding='utf-8'))
    lines = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for line in ''.join(cell['source']).splitlines():
                if not line.lstrip().startswith(('%', '!')):
                    lines.append(line.replace('display(', 'print('))
            lines.append('')
    TMP.write_text('\n'.join(lines), encoding='utf-8')
    return TMP


def main():
    scripts = [p for p in sorted(SRC.glob('*.py')) + sorted(SRC.glob('*.ipynb'))
               if p.name not in SKIP and p != TMP]
    env = dict(os.environ, PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')  # Agg: plt.show() не останавливает прогон
    REPORT.parent.mkdir(exist_ok=True)

    summary = []
    with open(REPORT, 'w', encoding='utf-8') as report:
        report.write(f'Общий прогон: {datetime.now():%d.%m.%Y %H:%M}\n')
        report.write(f'Скриптов: {len(scripts)}\n\n')

        for script in scripts:
            text = script.read_text(encoding='utf-8', errors='replace')
            found = [word for word in FORBIDDEN if word in text]

            print(f'Запускаю {script.name} ...', flush=True)
            start = time.perf_counter()
            try:
                target = notebook_to_script(script) if script.suffix == '.ipynb' else script
                res = subprocess.run(
                    [sys.executable, target.name], cwd=SRC, env=env,
                    capture_output=True, text=True, encoding='utf-8', errors='replace',
                    timeout=TIMEOUT_S,
                )
                status = 'OK' if res.returncode == 0 else 'ОШИБКА'
                out, err = res.stdout, res.stderr
            except subprocess.TimeoutExpired:
                status, out, err = 'НЕ УСПЕЛ', '', f'Остановлен через {TIMEOUT_S} с'
            seconds = time.perf_counter() - start
            summary.append((script.name, status, seconds, found))

            report.write('=' * 70 + '\n')
            report.write(f'{script.name}: {status}, {seconds:.0f} с\n')
            if found:
                report.write(f'ВНИМАНИЕ: в скрипте есть {", ".join(found)} — деление на обучение '
                             f'и проверку должно быть только из common.split()\n')
            report.write('-' * 70 + '\n')
            report.write(out)
            if status != 'OK' and err:
                report.write('\n--- ошибка ---\n' + err)
            report.write('\n')

        TMP.unlink(missing_ok=True)
        report.write('=' * 70 + '\nИтог\n')
        for name, status, seconds, found in summary:
            note = f'  внимание: {", ".join(found)}' if found else ''
            report.write(f'{name:40s} {status:9s} {seconds:5.0f} с{note}\n')

    print('\nИтог:')
    for name, status, seconds, found in summary:
        note = f'  внимание: {", ".join(found)}' if found else ''
        print(f'  {name:40s} {status:9s} {seconds:5.0f} с{note}')
    print(f'\nВесь вывод: reports/{REPORT.name}')


if __name__ == '__main__':
    main()
