/*
* particle_filter.cpp
*
*  Created on: Dec 12, 2016
*      Author: Tiffany Huang
*/

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 20;
	default_random_engine gen;

	//gaussian distribution for GPS 'x' postion
	normal_distribution<double> dist_x(x, std[0]);
	//gaussian distribution for GPS 'y' postion
	normal_distribution<double> dist_y(y, std[1]);
	//gaussian distribution for theta
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i=0; i< num_particles; i++){
		Particle particle;
		particle.id = i;

		//sampling from the normal destribution
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);

		// initialize particles weights
		particle.weight = 1.0;

		particles.push_back(particle);
		weights.push_back(particle.weight);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	for(int i=0; i< num_particles; i++){
		double particle_x = particles[i].x;
		double particle_y = particles[i].y;
		double particle_theta = particles[i].theta;

		if (fabs(yaw_rate) < 0.0001) {	// zero yaw rate
			particle_x += velocity * cos(particle_theta) * delta_t;
			particle_y += velocity * sin(particle_theta) * delta_t;
		} else {	// non zero yaw_rate
			particle_x += (velocity/yaw_rate) * (sin(particle_theta + (yaw_rate * delta_t)) - sin(particle_theta));
			particle_y += (velocity/yaw_rate) * (cos(particle_theta) - cos(particle_theta + (yaw_rate * delta_t)));
			particle_theta += (yaw_rate * delta_t);
		}

		normal_distribution<double> dist_x(particle_x, std_pos[0]);
		normal_distribution<double> dist_y(particle_y, std_pos[1]);
		normal_distribution<double> dist_theta(particle_theta, std_pos[2]);

		// adding guassian noice
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);

	}
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations, double sensor_range) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for (uint8_t i = 0; i < observations.size(); i++) {
		// maximum distance possible is the range of sensor.
		// first set lowest_dist to the maximum possible distance which will be updated later.
		double lowest_dist = sensor_range;
		int closest_landmark_id = -1;

		for (uint8_t j = 0; j < predicted.size(); j++) {
			// use predefined distance fucntion to find the euclidian distance between the predicted and observed
			double current_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if (current_dist < lowest_dist) {
				lowest_dist = current_dist;
				closest_landmark_id = predicted[j].id;
			}
		}
		observations[i].id = closest_landmark_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
		// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
		//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
		// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
		//   according to the MAP'S coordinate system. You will need to transform between the two systems.
		//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
		//   The following is a good resource for the theory:
		//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
		//   and the following is a good resource for the actual equation to implement (look at equation
		//   3.33
		//   http://planning.cs.uiuc.edu/node99.html


		double cumilative_weight = 0.0;

		for (uint8_t i = 0; i < num_particles; i++) {
			double particle_x = particles[i].x;
			double particle_y = particles[i].y;
			double particle_theta = particles[i].theta;

			// Transform observations from vehicle co-ordinates system to map co-ordinates
			vector<LandmarkObs> observations_in_map_co;
			for (uint8_t j = 0; j < observations.size(); j++) {
				LandmarkObs transformed_observations;
				transformed_observations.id = j;
				transformed_observations.x = particle_x + (cos(particle_theta) * observations[j].x) - (sin(particle_theta) * observations[j].y);
				transformed_observations.y = particle_y + (sin(particle_theta) * observations[j].x) + (cos(particle_theta) * observations[j].y);
				observations_in_map_co.push_back(transformed_observations);
			}

			// Now we filter out map_landmarks which are not in sensor_range and push the
			// remaining to predicted_landmarks
			vector<LandmarkObs> predicted_landmarks;

			for (uint8_t k = 0; k < map_landmarks.landmark_list.size(); k++) {
				Map::single_landmark_s landmark = map_landmarks.landmark_list[k];

				if ((fabs((particle_x - landmark.x_f)) <= sensor_range) && (fabs((particle_y - landmark.y_f)) <= sensor_range)) {
					predicted_landmarks.push_back(LandmarkObs {landmark.id_i, landmark.x_f, landmark.y_f});
				}
			}

			// associate observations to predicted_landmarks using nearest neighbor algorithm
			dataAssociation(predicted_landmarks, observations_in_map_co, sensor_range);

			// calculate the weight of each particle using Multivariate Gaussian distribution
			//Start with weight of particle = 1.0
			particles[i].weight = 1.0;

			double sigma_x = std_landmark[0];
			double sigma_y = std_landmark[1];
			double sigma_x_squared = pow(sigma_x, 2);
			double sigma_y_squared = pow(sigma_y, 2);
			double normalizer = (1.0/(2.0 * M_PI * sigma_x * sigma_y));
			uint8_t k, l;

			// calculate the weight of particle using multivariate Gaussian probability function
			for (k = 0; k < observations_in_map_co.size(); k++) {
				double trans_obs_x = observations_in_map_co[k].x;
				double trans_obs_y = observations_in_map_co[k].y;
				double trans_obs_id = observations_in_map_co[k].id;
				double multi_prob = 1.0;

				for (l = 0; l < predicted_landmarks.size(); l++) {
					double pred_landmark_x = predicted_landmarks[l].x;
					double pred_landmark_y = predicted_landmarks[l].y;
					double pred_landmark_id = predicted_landmarks[l].id;

					if (trans_obs_id == pred_landmark_id) {
						double exponent = exp(-1.0 * ((pow((trans_obs_x - pred_landmark_x), 2)/(2.0 * sigma_x_squared)) + (pow((trans_obs_y - pred_landmark_y), 2)/(2.0 * sigma_y_squared))));
						multi_prob = normalizer * exponent;
						particles[i].weight *= multi_prob;
					}
				}
			}
			cumilative_weight += particles[i].weight;
		}

		// normalize the weights of all particles by dividing my cumilative_weight
		for (uint8_t i = 0; i < particles.size(); i++) {
			particles[i].weight /= cumilative_weight;
			weights[i] = particles[i].weight;
		}

	}

	void ParticleFilter::resample() {
		// TODO: Resample particles with replacement with probability proportional to their weight.
		// NOTE: You may find std::discrete_distribution helpful here.
		//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
		default_random_engine gen;

		// Take a discrete distribution with pmf equal to weights
		discrete_distribution<> weights_pmf(weights.begin(), weights.end());
		// initialise new particle array
		vector<Particle> newParticles;
		// resample particles
		for (uint8_t i = 0; i < num_particles; ++i){
			newParticles.push_back(particles[weights_pmf(gen)]);
		}

		particles = newParticles;
	}

	Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
		const std::vector<double>& sense_x, const std::vector<double>& sense_y)
		{
			//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
			// associations: The landmark id that goes along with each listed association
			// sense_x: the associations x mapping already converted to world coordinates
			// sense_y: the associations y mapping already converted to world coordinates

			particle.associations.clear();
			particle.sense_x.clear();
			particle.sense_y.clear();

			particle.associations= associations;
			particle.sense_x = sense_x;
			particle.sense_y = sense_y;

			return particle;
		}

		string ParticleFilter::getAssociations(Particle best)
		{
			vector<int> v = best.associations;
			stringstream ss;
			copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
			string s = ss.str();
			s = s.substr(0, s.length()-1);  // get rid of the trailing space
			return s;
		}
		string ParticleFilter::getSenseX(Particle best)
		{
			vector<double> v = best.sense_x;
			stringstream ss;
			copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
			string s = ss.str();
			s = s.substr(0, s.length()-1);  // get rid of the trailing space
			return s;
		}
		string ParticleFilter::getSenseY(Particle best)
		{
			vector<double> v = best.sense_y;
			stringstream ss;
			copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
			string s = ss.str();
			s = s.substr(0, s.length()-1);  // get rid of the trailing space
			return s;
		}
