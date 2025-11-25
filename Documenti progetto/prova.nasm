%include "sseutils32.nasm"
section .data
	align 16
	v	dd	2.1, 3.5, 4.3, 6.8
section .bss
	alignb 16
	w resd 4
section .text
	global main
main:
	start
	MOVAPS	XMM0, [v]
	MOVAPS	[w], XMM0
	printps	w, 1
	stop
