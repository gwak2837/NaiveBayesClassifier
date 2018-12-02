#include <iostream>
#include <cmath>
#include "NaiveBayesClassifier.h"

double l1(double x) {
	return -pow(x, 2) + 1;
}

double l2(double x) {
	return -pow(x - 0.5, 2) + 1;
}

int main()
{	
	// define required value of classifier
	vector<function<double(double)>> likelyhood = { l1, l2 };  // P(x|yi) 
	vector<string> label;       // yi
	vector<double> prior;       // P(yi)
	unsigned int n = 2;

	label.reserve(n);
	label.push_back("salmon");
	label.push_back("seabass");
	
	prior.reserve(n);
	prior.push_back(2.0 / 3.0);
	prior.push_back(1.0 / 3.0);

	// create classifier
	NaiveBayesClassifier * nbc = new NaiveBayesClassifier(likelyhood, prior, label);
	
	// print result
	for (double i = -2.0; i < 2.0; i += 0.1) {
		cout << "Posterior: " << nbc->MAP_of(i) << ", Label: " << nbc->label_of_MAP_of(i) << "\n\n";
	}

	return 0;
}