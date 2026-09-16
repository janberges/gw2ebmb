.PHONY: all

all: fig1.pdf fig2.pdf fig3.pdf fig4.pdf fig5.pdf fig6.pdf

CMD = python3

%.pdf: %.py
	$(CMD) $<

fig2.pdf: fig2ef.txt
fig4.pdf: fig4a.txt fig4b.txt
fig6.pdf: fig6def.txt
