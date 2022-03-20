hw2: MMult1_omp.cpp val_test01.cpp val_test02.cpp omp_bug2.c omp_bug3.c omp_bug4.c omp_bug5.c omp_bug6.c jacobi2D-omp.cpp gs2D-omp.cpp 
	g++ -std=c++11 -fopenmp -O3 -march=native MMult1_omp.cpp -o MMult1
	g++ -std=c++11 val_test01.cpp -o val_test01_solved
	g++ -std=c++11 val_test02.cpp -o val_test02_solved
	g++ -std=c++11 -fopenmp omp_bug2.c -o omp_solved2
	g++ -std=c++11 -fopenmp omp_bug3.c -o omp_solved3
	g++ -std=c++11 -fopenmp omp_bug4.c -o omp_solved4
	g++ -std=c++11 -fopenmp omp_bug5.c -o omp_solved5
	g++ -std=c++11 -fopenmp omp_bug6.c -o omp_solved6
	g++ -std=c++11 -fopenmp jacobi2D-omp.cpp -o jacobi2D-omp
	g++ -std=c++11 -fopenmp gs2D-omp.cpp -o gs2D-omp

