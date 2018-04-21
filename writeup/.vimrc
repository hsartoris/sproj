let g:latex_build_dir = "build"

command LoadAll :call LoadChapters()

function! LoadChapters()
	tabe chapters/chapter1.tex
	tabe chapters/chapter2.tex
	tabe chapters/chapter3.tex
	tabn
endfunction
