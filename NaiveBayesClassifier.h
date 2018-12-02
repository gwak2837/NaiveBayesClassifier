#pragma once

#ifndef __NAIVE_BAYES_CLASSIFIER_H_
#define __NAIVE_BAYES_CLASSIFIER_H_
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <functional>
#include <cstdint>
using namespace std;

// 수정사항: 
// (진행중) max_element and label need to fix floating point error
// (진행중) prior의 합이 1인지 검사, P(y1) + ... + P(yn) = 1, 부동소수점 오차 처리법?
// vector<string> likelyhood; push "x^2 + 0.5"; push "(x-2)^2 + 0.5"; // string -> equation 변환 필요

class NaiveBayesClassifier {
	vector<function<double(double)>> likelyhood;  // P(x|yi)
	vector<double> prior;                         // P(yi)
	vector<string> label;                         // yi
	vector<double> posterior;                     // P(yi|x)
	unsigned int n;                               // the maximun number of i

	double observation;                           // x
	bool first = true;                            // chech whether first or not

public:
	NaiveBayesClassifier(vector<function<double(double)>> likelyhood, vector<double> prior, vector<string> label) {
		if (prior.size() != label.size()) {
			cout << "error: The number of elements of prior and label are different\n";
			exit(-1);
		}
		else if (prior.size() != likelyhood.size()) {
			cout << "error: The number of elements of prior and likelyhood are different\n";
			exit(-1);
		}
		else if (label.size() != likelyhood.size()) {
			cout << "error: The number of elements of label and likelyhood are different\n";
			exit(-1);
		}
		else if (1) {
			// (진행중) prior의 합이 1인지 검사, P(y1) + ... + P(yn) = 1, 부동소수점 오차 처리법?
			for_each(prior.begin(), prior.end(), [](int &n) { n++; });
		}
		else {
			this->likelyhood = likelyhood;
			this->prior = prior;
			this->label = label;
			this->n = prior.size();
		}
	}
	


	// compare two double values considering floating point error
	// if x == y, return 0, if x > y, return 1, if x < y return -1
	int cmp_abs_and_ULPs(double x, double y, double abs_tol = (1.0e-8), int ulps_tol = 4)
	{
		double diff = x - y;
		if (fabs(diff) <= abs_tol)
			return 0;

		int64_t nx = *((int64_t*)&x);
		int64_t ny = *((int64_t*)&y);

		if ((nx & 0x8000000000000000) != (ny & 0x8000000000000000))
			return (diff > 0) ? 1 : -1;

		int64_t ulpsDiff = nx - ny;
		if ((ulpsDiff >= 0 ? ulpsDiff : -ulpsDiff) <= ulps_tol)
			return 0;

		return (diff > 0) ? 1 : -1;
	}

	// every probability should be 0 or over and 1 or below(0 <= P(x) <= 1)
	double nonnegativity(double probability, int index, double observation) {
		if (cmp_abs_and_ULPs(probability, 1.0) == 1) {
			cout << "warning: codomain of likelyhood[" << index << "](" << observation << ") is bigger than one\n";
			return 1.0;
		}
		else if (cmp_abs_and_ULPs(0.0, probability) == 1) {
			cout << "warning: codomain of likelyhood[" << index << "](" << observation << ") is smaller than zero\n";
			return 0.0;
		}
		else
			return probability;
	}

	double nonnegativity(double probability) {
		if (cmp_abs_and_ULPs(probability, 1.0) == 1)
			return 1.0;
		else if (cmp_abs_and_ULPs(0.0, probability) == 1)
			return 0.0;
		else
			return probability;
	}

	// Bayes Theorem: P(yi|x) = P(x|yi)P(yi) / P(x|y1) + ... + P(x|yn)
	// Return: if divide by zero occurs, return false, else return true
	bool bayes_theorem(double observation) {
		if (first) {
			first = false;
			this->observation = observation;
			goto update_posterior;
		}

		// Reuse the previous posterior if previous observation is same as given observation
		if (!cmp_abs_and_ULPs(this->observation, observation)) {
			if (!posterior.size())
				return false;
			else
				return true;
		}

		// If previous observation and this observation are different, 
		// clear posterior and update posterior and observation
		this->observation = observation;
		posterior.clear();

	// update posterior with newly given observation
	update_posterior:
		unsigned int i;
		double evidence = 0.0; // P(x)
		
		// calculate P(x) by Total Probability Theorem
		for (i = 0; i < n; i++) {
			double result = nonnegativity(likelyhood[i](observation), i, observation);
			evidence += result * prior[i];
		}

		// handle exception of divide by zero
		if (!evidence)
			return false;
			
		// calculate P(yi intersect x) by Multiplication Rule
		for (i = 0; i < n; i++) {
			double result = nonnegativity(likelyhood[i](observation));
			posterior.push_back(likelyhood[i](observation) * prior[i] / evidence);
		}
		return true;
	}

	// Maximum A Posterior
	// Return: the maximum value of posterior of given observation
	double MAP_of(double observation) {
		if (!bayes_theorem(observation))
			return 0.0; // evidence is zero(codomain of likelyhood is undefined?)
			
		// get the maximum value in P(yi|x) and return it
		return *max_element(posterior.begin(), posterior.end());
	}

	// label of the Maximun A Posterior
	// Return: label of the MAP of given observation
	string label_of_MAP_of(double observation) {
		if (!bayes_theorem(observation))
			return "error: evidence is zero\n";

		int index;
		double max = 0.0;

		// get the maximum value of P(yi|x) and return yi
		for (unsigned int i = 0; i < n; i++) {
			if (cmp_abs_and_ULPs(posterior[i], max) == 1) {
				max = posterior[i];
				index = i;
			}
		}

		return label[index];
	}
};
#endif