let g:latex_build_dir = "build"

command LoadAll :call LoadChapters()

function! LoadChapters()
	tabe chapters/bg.tex
	tabe chapters/model.tex
	tabe chapters/train.tex
	tabe chapters/results.tex
	tabn
endfunction
