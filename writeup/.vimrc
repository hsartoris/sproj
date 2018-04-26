let g:latex_build_dir = "build"

command LoadAll :call LoadChapters()

function! LoadChapters()
	tabe chapters/chapter_bg.tex
	tabe chapters/chapter_model.tex
	tabe chapters/chapter_results.tex
	tabn
endfunction
