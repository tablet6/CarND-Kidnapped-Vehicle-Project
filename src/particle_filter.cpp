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
  
  num_particles = 150;
  default_random_engine gen;
  
  // Create a normal (Gaussian) distribution
  normal_distribution<double> dist_theta(theta, std[2]);
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);

  for (int i=0; i<num_particles; i++)
  {
    Particle p;
    p.id     = i;
    p.x      = dist_x(gen);
    p.y      = dist_y(gen);
    p.theta  = dist_theta(gen);
    p.weight = 1.0;
    
    particles.push_back(p);
    weights.push_back(1.0);
    
//    std::cout<<"p.x: "<<p.x<<" p.y: "<<p.y<<" p.theta: "<<p.theta<<"\n";
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  

  default_random_engine gen;
  normal_distribution<double> dist_vel(0.0, velocity);
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);
  
  for (int i=0; i<num_particles; i++) {
    double yaw =  particles[i].theta;

    if (fabs(yaw_rate) < 0.0001) {
      particles[i].x += (velocity * delta_t * cos(yaw));
      particles[i].y += (velocity * delta_t * sin(yaw));
    }
    else {
      particles[i].x     += ((velocity / yaw_rate) * (sin(yaw + yaw_rate * delta_t) - sin(yaw)));
      particles[i].y     += ((velocity / yaw_rate) * (cos(yaw) - cos(yaw + yaw_rate * delta_t)));
      particles[i].theta += (yaw_rate * delta_t);
    }
    
    //Noise
    particles[i].x     += dist_x(gen);
    particles[i].y     += dist_y(gen);
    particles[i].theta += dist_theta(gen);
    velocity           += dist_vel(gen);
  }
}

bool ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, LandmarkObs& observation) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  
  double min_distance = std::numeric_limits<double>::max();
  int save_id = -1;
  
  for (int j=0; j<predicted.size(); j++) {
    
    double distance = dist(observation.x, observation.y, predicted[j].x, predicted[j].y);
    if (distance < min_distance) {
      min_distance = distance;
      save_id = j;
    }
  }
  
  if (save_id == -1)
    return false;
  
  observation.id = save_id;
  return true;
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
  
  
  for (int i=0; i<num_particles; i++) {
    
    double x_part     = particles[i].x;
    double y_part     = particles[i].y;
    double theta      = particles[i].theta;
    double weight     = 1.0;
    double sigma_x    = std_landmark[0];
    double sigma_y    = std_landmark[1];
    double gauss_norm = (1/(2 * M_PI * sigma_x * sigma_y));

    
    for (int j=0; j<observations.size(); j++) {
      
      // Transform: Transform car sensor landmark observations from the car coordinate system, to the map coordinate system,
      double x_map = x_part + (cos(theta) * observations[j].x) - (sin(theta) * observations[j].y);
      double y_map = y_part + (sin(theta) * observations[j].x) + (cos(theta) * observations[j].y);
      
      LandmarkObs obs_transformed;
      obs_transformed.id = observations[j].id;
      obs_transformed.x  = x_map;
      obs_transformed.y  = y_map;
      
      // Find landmarks near this particle within the sensor range
      std::vector<LandmarkObs> predicted_landmarksObs;
      
      for (int k=0; k<map_landmarks.landmark_list.size(); k++) {
        int   landmark_id = map_landmarks.landmark_list[k].id_i;
        float landmark_x  = map_landmarks.landmark_list[k].x_f;
        float landmark_y  = map_landmarks.landmark_list[k].y_f;
        
        double distance = dist(x_part, y_part, landmark_x, landmark_y);
        if (distance <= sensor_range) {
          
          LandmarkObs obs_predicted;
          obs_predicted.id = landmark_id;
          obs_predicted.x  = landmark_x;
          obs_predicted.y  = landmark_y;
          
          predicted_landmarksObs.push_back(obs_predicted);
        }
      }
      
      // Associate: Associating these transformed observations with the nearest landmark on the map.
      if (!dataAssociation(predicted_landmarksObs, obs_transformed))
        continue;

      // Update Weights: Updating particle weight by applying the multivariate Gaussian probability density function for each measurement.
      LandmarkObs mu  = predicted_landmarksObs[obs_transformed.id];
      
      double x_obs = obs_transformed.x;
      double y_obs = obs_transformed.y;
      double mu_x  = mu.x;
      double mu_y  = mu.y;
      
      double exponent = ((x_obs - mu_x)*(x_obs - mu_x))/(2 * sigma_x * sigma_x) + ((y_obs - mu_y)*(y_obs - mu_y))/(2 * sigma_y * sigma_y);
      weight *= gauss_norm * exp(-exponent);
    }
    
    particles[i].weight = weight;
    weights[i]          = weight; //Need it for resampling
  }
}

void ParticleFilter::resample() {
  
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
  default_random_engine gen;
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  std::vector<Particle> resampled_particles;
  
  for (int i=0; i<num_particles; i++) {
    int index = dist(gen);
    resampled_particles.push_back(particles[index]);
  }

  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
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
