(TeX-add-style-hook
 "readme"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "graphicx"
    "verbatim")
   (LaTeX-add-labels
    "fig:basic"))
 :latex)

