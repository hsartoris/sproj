let g:latex_build_dir = "build"
let g:vimtex_quickfix_enabled = 0

let g:vimtex_latexmk_options = '-pdf -bibtex -file-line-error'

command LoadAll :call LoadChapters()

function! LoadChapters()
	tabe chapters/bg.tex
	tabe chapters/model.tex
	tabe chapters/train.tex
	tabe chapters/results.tex
	tabe chapters/discussion.tex
	tabn
endfunction
