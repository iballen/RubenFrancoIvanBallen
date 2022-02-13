#include <iostream>
#include <fstream>
#include <math.h>


double f(double x){
	return exp(-1*(x*x));
}

double derivada_central(double x, double h){
	return (f(x+h/2)-f(x-h/2))/h;
}

int main(){
	std::ofstream *File; //Apuntador
	
	File = new std::ofstream[2]; //Reservamos memoria
	
	File[0].open("Derivada.txt", std::ofstream::trunc); //Archivo abierto y listo para escribir
	
	double x = -20.0;
	float h = 0.01;
	
	while (x <= 20){
		File[0] << x << " " << derivada_central(x,h) << std::endl;
		x = x+h;
	}
}