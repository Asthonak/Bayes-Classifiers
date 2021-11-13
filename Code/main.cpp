#include <iostream>
#include "boxmuller.cpp"
#include "Classifiers.cpp"
#include <fstream>
#include <cmath>
#include <math.h>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

/**Function that calculates the Bhattacharyya bound**/
float calculateBhattacharyya(float beta, Vector2f mu_1, Vector2f mu_2, Matrix2f sigma_1, Matrix2f sigma_2)
{
	float k_beta = (beta*(1-beta))/2.0;
	k_beta *= (mu_1 - mu_2).transpose() * ((1-beta)*sigma_1 + (beta)*sigma_2).inverse() * (mu_1-mu_2);
	k_beta += 0.5 * log(((1-beta)*sigma_1 + (beta)*sigma_2).determinant() / (pow(sigma_1.determinant(), 1-beta) 
	       * pow(sigma_2.determinant(), beta)));

	float bhattacharyya_error = exp(-1.0*k_beta);

	return bhattacharyya_error;
}

int main()
{
	vector<Vector2f> distribution1;
	vector<Vector2f> distribution2;

	Vector2f mu_1;
	Matrix2f sigma_1;
	Vector2f mu_2;
	Matrix2f sigma_2;

	/*Problem 1*/

	mu_1 << 1, 1;
	sigma_1 << 1, 0, 
		   0, 1;
	mu_2 << 4, 4;
	sigma_2 << 1, 0, 
		   0, 1;

	// generate the samples
	for(int i = 0; i < 100000; i++)
	{
		distribution1.push_back(Vector2f(box_muller(mu_1(0,0), sigma_1(0,0)), box_muller(mu_1(1,0), sigma_1(1,1))));
	}

	for(int i = 0; i < 100000; i++)
	{
		distribution2.push_back(Vector2f(box_muller(mu_2(0,0), sigma_2(0,0)), box_muller(mu_2(1,0), sigma_2(1,1))));
	}

	// log the samples on output files

	ofstream outputFile;

	outputFile.open("p1_data1x.txt");
	for(int i=0;i<distribution1.size();i++)
	{
		outputFile << distribution1[i](0) << endl;
	}
	outputFile.close();

	outputFile.open("p1_data2x.txt");

	for(int i=0;i<distribution2.size();i++)
	{
		outputFile << distribution2[i](0) << endl;
	}
	outputFile.close();

	outputFile.open("p1_data1y.txt");
	for(int i=0;i<distribution1.size();i++)
	{
		outputFile << distribution1[i](1) << endl;
	}
	outputFile.close();

	outputFile.open("p1_data2y.txt");
	for(int i=0;i<distribution2.size();i++)
	{
		outputFile << distribution2[i](1) << endl;
	}
	outputFile.close();

	cout << "Problem 1a: " << endl;

	// array to hold decision made for the classifer
	bool p1_a_decision[200000];

	// prior probabilities for 1a
	float prior_prob_1 = 0.5;
	float prior_prob_2 = 0.5;

	// classify the samples using case 1
	for (int i = 0, j = 100000; i < 100000; i++, j++)
	{
			float gi = discriminant_case_1(distribution1[i], mu_1, sigma_1(0,0), prior_prob_1);
			float gj = discriminant_case_1(distribution1[i], mu_2, sigma_2(0,0), prior_prob_2);
//			if(i == 0)
//				cout << "1a \t" << gi << "\t" << gj << endl;	
			if (gi - gj > 0)
			{
				p1_a_decision[i] = 1;
			}
			else
			{
				p1_a_decision[i] = 0;
			}

			gi = discriminant_case_1(distribution2[i], mu_1, sigma_1(0,0), prior_prob_1);
			gj = discriminant_case_1(distribution2[i], mu_2, sigma_2(0,0), prior_prob_2);
//			if(i == 0)
//				cout << "1a \t" << gi << "\t" << gj << endl;
			if (gi - gj > 0)
			{
				p1_a_decision[j] = 1;
			}
			else
			{
				p1_a_decision[j] = 0;
			}
	}

	// calculate the number of misclassifications made and log decisions to an output file
	outputFile.open("p1_a_decisions.txt");
	int wrong1 = 0;
	int wrong2 = 0;
	for (int i = 0; i < 200000; i++)
	{
		outputFile << p1_a_decision[i] << endl;
		if (i < 100000)
		{
			if (p1_a_decision[i] == 0)
			{
				wrong1++;
			}
		}
		else
		{
			if (p1_a_decision[i] == 1)
			{
				wrong2++;
			}
		}
	}
	outputFile.close();

	cout << "number wrong from class 1: " << wrong1 << endl;
	cout << "number wrong from class 2: " << wrong2 << endl;
	cout << "total number wrong: " << wrong1 + wrong2 << endl;

	// calculate the Bhattacharyya bound
	float beta = 0.5;

	float bhattacharyya = calculateBhattacharyya(beta, mu_1, mu_2, sigma_1, sigma_2);

	cout << "Bhattacharyya Bound (Beta = " << beta << "): " << bhattacharyya << endl;


	cout << endl << "Problem 1b: " << endl;

	// array to hold decisions made by the classifer
	bool p1_b_decision[200000];

	prior_prob_1 = 0.2;
	prior_prob_2 = 0.8;

	// classify the samples using case 1
	for (int i = 0, j = 100000; i < 100000; i++, j++)
	{
			float gi = discriminant_case_1(distribution1[i], mu_1, sigma_1(0,0), prior_prob_1);
			float gj = discriminant_case_1(distribution1[i], mu_2, sigma_2(0,0), prior_prob_2);
//			if(i == 0)
//				cout << "1b \t" << gi << "\t" << gj << endl;
			if (gi - gj > 0)
			{
				p1_b_decision[i] = 1;
			}
			else
			{
				p1_b_decision[i] = 0;
			}

			gi = discriminant_case_1(distribution2[i], mu_1, sigma_1(0,0), prior_prob_1);
			gj = discriminant_case_1(distribution2[i], mu_2, sigma_2(0,0), prior_prob_2);
//			if(i == 0)
//				cout << "1b \t" << gi << "\t" << gj << endl;
			if (gi - gj > 0)
			{
				p1_b_decision[j] = 1;
			}
			else
			{
				p1_b_decision[j] = 0;
			}
	}

	// calculate the number of misclassifications made and log decisions to an output file
	outputFile.open("p1_b_decisions.txt");
	wrong1 = 0;
	wrong2 = 0;
	for (int i = 0; i < 200000; i++)
	{
		outputFile << p1_b_decision[i] << endl;
		if (i < 100000)
		{
			if (p1_b_decision[i] == 0)
			{
				wrong1++;
			}
		}
		else
		{
			if (p1_b_decision[i] == 1)
			{
				wrong2++;
			}
		}
	}
	outputFile.close();

	cout << "number wrong from class 1: " << wrong1 << endl;
	cout << "number wrong from class 2: " << wrong2 << endl;
	cout << "total number wrong: " << wrong1 + wrong2 << endl;

	// calculate the Bhattacharyya bound
	bhattacharyya = calculateBhattacharyya(beta, mu_1, mu_2, sigma_1, sigma_2);

	cout << "Bhattacharyya Bound (Beta = " << beta << "): " << bhattacharyya << endl;

	/*Problem 2*/

	mu_1 << 1, 1;
	sigma_1 << 1, 0, 
		   0, 1;
	mu_2 << 4, 4;
	sigma_2 << 4, 0, 
		   0, 8;

	// generate the samples

	distribution1.clear();
	distribution2.clear();
	for(int i = 0; i < 100000; i++)
	{
		distribution1.push_back(Vector2f(box_muller(mu_1(0,0), sigma_1(0,0)), box_muller(mu_1(1,0), sigma_1(1,1))));
	}

	for(int i = 0; i < 100000; i++)
	{
		distribution2.push_back(Vector2f(box_muller(mu_2(0,0), sigma_2(0,0)), box_muller(mu_2(1,0), sigma_2(1,1))));
	}

	// log the samples on output files

	outputFile.open("p2_data1x.txt");
	for(int i=0;i<distribution1.size();i++)
	{
		outputFile << distribution1[i](0) << endl;
	}
	outputFile.close();

	outputFile.open("p2_data2x.txt");
	for(int i=0;i<distribution2.size();i++)
	{
		outputFile << distribution2[i](0) << endl;
	}
	outputFile.close();

	outputFile.open("p2_data1y.txt");
	for(int i=0;i<distribution1.size();i++)
	{
		outputFile << distribution1[i](1) << endl;
	}
	outputFile.close();

	outputFile.open("p2_data2y.txt");
	for(int i=0;i<distribution2.size();i++)
	{
		outputFile << distribution2[i](1) << endl;
	}
	outputFile.close();

	cout << endl << "Problem 2a: " << endl;

	// array to hold decisions made by the classifer
	bool p2_a_decision[200000];

	prior_prob_1 = 0.5;
	prior_prob_2 = 0.5;

	// classify the samples using case 3
	for (int i = 0, j = 100000; i < 100000; i++, j++)
	{
			float gi = discriminant_case_3(distribution1[i], mu_1, sigma_1, prior_prob_1);
			float gj = discriminant_case_3(distribution1[i], mu_2, sigma_2, prior_prob_2);
			if (gi - gj > 0)
			{
				p2_a_decision[i] = 1;
			}
			else
			{
				p2_a_decision[i] = 0;
			}

			gi = discriminant_case_3(distribution2[i], mu_1, sigma_1, prior_prob_1);
			gj = discriminant_case_3(distribution2[i], mu_2, sigma_2, prior_prob_2);
			if (gi - gj > 0)
			{
				p2_a_decision[j] = 1;
			}
			else
			{
				p2_a_decision[j] = 0;
			}
	}

	// calculate the number of misclassifications made and log decisions to an output file
	outputFile.open("p2_a_decisions.txt");
	wrong1 = 0;
	wrong2 = 0;
	for (int i = 0; i < 200000; i++)
	{
		outputFile << p2_a_decision[i] << endl;
		if (i < 100000)
		{
			if (p2_a_decision[i] == 0)
			{
				wrong1++;
			}
		}
		else
		{
			if (p2_a_decision[i] == 1)
			{
				wrong2++;
			}
		}
	}
	outputFile.close();

	cout << "number wrong from class 1: " << wrong1 << endl;
	cout << "number wrong from class 2: " << wrong2 << endl;
	cout << "total number wrong: " << wrong1 + wrong2 << endl;

	// calculate the Bhattacharyya bound
	beta = 0.5;

	bhattacharyya = calculateBhattacharyya(beta, mu_1, mu_2, sigma_1, sigma_2);

	cout << "Bhattacharyya Bound (Beta = " << beta << "): " << bhattacharyya << endl;

	cout << endl << "Problem 2b: " << endl;

	// array to hold decisions made by the classifer
	bool p2_b_decision[200000];

	prior_prob_1 = 0.2;
	prior_prob_2 = 0.8;

	// classify the samples using case 3
	for (int i = 0, j = 100000; i < 100000; i++, j++)
	{
			float gi = discriminant_case_3(distribution1[i], mu_1, sigma_1, prior_prob_1);
			float gj = discriminant_case_3(distribution1[i], mu_2, sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p2_b_decision[i] = 1;
			}
			else
			{
				p2_b_decision[i] = 0;
			}

			gi = discriminant_case_3(distribution2[i], mu_1, sigma_1, prior_prob_1);
			gj = discriminant_case_3(distribution2[i], mu_2, sigma_2, prior_prob_2);

			if (gi - gj > 0)
			{
				p2_b_decision[j] = 1;
			}
			else
			{
				p2_b_decision[j] = 0;
			}
	}

	// calculate the number of misclassifications made and log decisions to an output file
	outputFile.open("p2_b_decisions.txt");
	wrong1 = 0;
	wrong2 = 0;
	for (int i = 0; i < 200000; i++)
	{
		outputFile << p2_b_decision[i] << endl;
		if (i < 100000)
		{
			if (p2_b_decision[i] == 0)
			{
				wrong1++;
			}
		}
		else
		{
			if (p2_b_decision[i] == 1)
			{
				wrong2++;
			}
		}
	}
	outputFile.close();

	cout << "number wrong from class 1: " << wrong1 << endl;
	cout << "number wrong from class 2: " << wrong2 << endl;
	cout << "total number wrong: " << wrong1 + wrong2 << endl;

	// calculate the Bhattacharyya bound
	bhattacharyya = calculateBhattacharyya(beta, mu_1, mu_2, sigma_1, sigma_2);

	cout << "Bhattacharyya Bound (Beta = " << beta << "): " << bhattacharyya << endl;

	/*Problem 3*/

	cout << endl << "Problem 3: " << endl;

	// the samples generated for problem 2 are also used for problem 3

	// array to hold decisions made by the classifer
	bool p3_decision[200000];

	// classify the samples using minimum distance case
	for (int i = 0, j = 100000; i < 100000; i++, j++)
	{
			float gi = discriminant_min_distance(distribution1[i], mu_1);
			float gj = discriminant_min_distance(distribution1[i], mu_2);
			if (gi - gj > 0)
			{
				p3_decision[i] = 1;
			}
			else
			{
				p3_decision[i] = 0;
			}

			gi = discriminant_min_distance(distribution2[i], mu_1);
			gj = discriminant_min_distance(distribution2[i], mu_2);
			if (gi - gj > 0)
			{
				p3_decision[j] = 1;
			}
			else
			{
				p3_decision[j] = 0;
			}

	}

	// calculate the number of misclassifications made and log decisions to an output file
	outputFile.open("p3_decision.txt");
	wrong1 = 0;
	wrong2 = 0;
	for (int i = 0; i < 200000; i++)
	{
		outputFile << p3_decision[i] << endl;
		if (i < 100000)
		{
			if (p3_decision[i] == 0)
			{
				wrong1++;
			}
		}
		else
		{
			if (p3_decision[i] == 1)
			{
				wrong2++;
			}
		}
	}
	outputFile.close();

	cout << "number wrong from class 1: " << wrong1 << endl;
	cout << "number wrong from class 2: " << wrong2 << endl;
	cout << "total number wrong: " << wrong1 + wrong2 << endl;

	// calculate the Bhattacharyya bound
	bhattacharyya = calculateBhattacharyya(beta, mu_1, mu_2, sigma_1, sigma_2);

	cout << "Bhattacharyya Bound (Beta = " << beta << "): " << bhattacharyya << endl;

}
