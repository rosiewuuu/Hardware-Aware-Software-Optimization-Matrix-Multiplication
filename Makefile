mm: mm.cpp
	g++ -O2 -mavx -mavx2 -fopenmp mm.cpp -o mm
clean:
	rm -rf mm *~
