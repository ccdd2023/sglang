# latexmkrc — sglang-kvflow reports
#
# Use xelatex via latexmk for the date-stamped progress report under reports/.
# xelatex is required because ctex's fandol fontset needs OTF support
# (FandolSong/FandolHei/FandolKai OTFs ship with ctex itself).
# Engines live at /home/gfy/texlive/2026/bin/x86_64-linux/ (not on PATH by default).
#
# Build:
#   export PATH=/home/gfy/texlive/2026/bin/x86_64-linux:$PATH
#   cd reports && make report

$xelatex  = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';
$pdf_mode = 1;
$out_dir  = 'reports/build';
$clean_ext = 'synctex.gz acn alg glg glo gls nav snm vrb run.xml bbl';