# Сборка диплома

Минимальный исходник LaTeX с нужными рисунками.

```bash
./build.sh
```

Если `latexmk` недоступен, можно собрать вручную:

```bash
pdflatex -interaction=nonstopmode -halt-on-error coursework.tex
biber coursework
pdflatex -interaction=nonstopmode -halt-on-error coursework.tex
pdflatex -interaction=nonstopmode -halt-on-error coursework.tex
```
